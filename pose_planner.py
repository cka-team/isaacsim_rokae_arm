# =============================================================================
# pose_planner.py — PoseIKPlanner（IK 路径规划器）
#
# 这是整个机器人控制堆栈的最上层接口，组合了：
#   + IKSolver         差分 IK（阻尼最小二乘）
#   + smoothstep5      五次光滑轨迹插值
#   + quaternion_slerp 姿态球面插值
#   + 关节限位保护     clip 输出以防超行程
#
# app0.py（以及将来的 app1.py）中所有机械臂运动都通过此类实现。
# =============================================================================

import math

from isaacsim.sensors.experimental.physics import ContactSensor
import numpy as np
from isaacsim.core.experimental.prims import RigidPrim

from .config import arm_config, grasp_config, gripper_config, ik_config
from .ik_solver import IKSolver
from .math_utils import (
    normalize_quaternion,
    orientation_error,
    quaternion_slerp,
    smoothstep5,
    to_numpy,
)
from .robot import GripperController, RobotArm
from .sim_app import SimulationManager


# =============================================================================
# PoseIKPlanner
#
# 对外接口（被 app0.py 调用）：
#   follow_pose_path()    → 笛卡尔空间路径规划（线性插值位置 + slerp 姿态）
#   actuate_gripper()     → 夹爪开合
#   set_home_pose_now()   → 瞬间跳转到 home 位姿
#   return_home()         → 平滑回到 home
#   get_pose()            → 读取当前 TCP 位姿
#   get_arm_positions()   → 读取当前关节角
#
# 内部方法：
#   command_pose()        → 单步 IK 闭环：误差 → 雅可比 → DLS → 下发 → 步进
#   _get_ee_jacobian()    → 从 Articulation 雅可比矩阵提取末端执行器的 6×7 子矩阵
#
# 数据流（以 follow_pose_path 为例）：
#
#   follow_pose_path(pos, ori, steps, settle)
#     │
#     ├─ path_steps 次:
#     │    phase = smoothstep5(step/steps)
#     │    waypoint = lerp(start, goal, phase)        // 位置线性插值
#     │    waypoint_ori = slerp(start, goal, phase)   // 姿态球面插值
#     │    command_pose(waypoint)                      // 一次 IK 闭环
#     │
#     └─ settle_steps 次:
#          command_pose(goal)                          // 在目标位姿收敛
#          break if (position_error < tolerance
#                    && rotation_error < tolerance)
# =============================================================================
class PoseIKPlanner:
    def __init__(self, robot: RobotArm, gripper: GripperController):
        """
        Args:
            robot:   RobotArm 实例，提供关节读写和雅可比
            gripper: GripperController 实例，用于开合夹爪
        """
        self._robot = robot
        self._gripper = gripper

        # 创建 IKSolver（阻尼最小二乘法，λ=0.05）
        self._ik = IKSolver(
            method="damped-least-squares",
            damping=0.05,
            scale=1.0,
        )

        # 通过 Singleton 获取 SimulationManager
        self._sim = SimulationManager.instance()

        # ---- 末端执行器设置 ------------------------------------------------
        # ee_link_index: 在 Articulation link 列表中的索引
        # ee_jacobian_index: 雅可比矩阵中的索引（通常 = link_index - 1）
        #                    因为 Articulation 的第一个 link 通常是基座，
        #                    其雅可比索引从 1 开始
        self.ee_link_index = robot.get_link_index(arm_config.end_effector_link)
        self.ee_jacobian_index = self.ee_link_index - 1
        self._ee_prim = RigidPrim(robot.link_paths[0][self.ee_link_index])
        print(f"[INFO] 末端执行器 link: {arm_config.end_effector_link} (index={self.ee_link_index})")

        # ---- 读取关节限位 --------------------------------------------------
        # 用于 command_pose() 中对 IK 输出做安全裁剪
        lower, upper = robot.get_dof_limits(dof_indices=robot.arm_indices)
        self.lower_limits = to_numpy(lower)[0].astype(np.float64)
        self.upper_limits = to_numpy(upper)[0].astype(np.float64)

    @property
    def robot(self) -> RobotArm:
        return self._robot

    @property
    def gripper(self) -> GripperController:
        return self._gripper

    # =========================================================================
    # 状态读取
    # =========================================================================

    def get_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """获取末端执行器（link7）当前世界位姿。
        
        Returns:
            (position (3,), orientation (4,) 归一化四元数 [w,x,y,z])
        """
        positions, orientations = self._ee_prim.get_world_poses()
        position = to_numpy(positions)[0].astype(np.float64)
        orientation = normalize_quaternion(
            to_numpy(orientations)[0].astype(np.float64)
        )
        return position, orientation

    def get_arm_positions(self) -> np.ndarray:
        """获取 7 个机械臂关节当前角度（弧度）。"""
        return self._robot.get_arm_positions()

    # =========================================================================
    # 内部：提取末端执行器雅可比
    # =========================================================================

    def _get_ee_jacobian(self) -> np.ndarray:
        """从 Articulation 全矩阵提取末端执行器的 6×7 雅可比。
        
        雅可比矩阵结构：J_world = [J_linear; J_angular] ∈ ℝ^(6×n_dof)
            J_linear  对应线速度（3 行）
            J_angular 对应角速度（3 行）
        
        Returns:
            (6, 7) numpy 数组
        """
        jacobians = to_numpy(self._robot.get_jacobian_matrices())
        ee_jacobian = jacobians[
            0, self.ee_jacobian_index, :, : arm_config.arm_dof_count
        ].astype(np.float64)
        return ee_jacobian

    # =========================================================================
    # 单步 IK 闭环
    # =========================================================================

    def command_pose(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        open_gripper: bool = True,
    ) -> tuple[float, float]:
        """单步差分 IK 命令：从当前位置向目标走一步。

        流程：
          1. 读取当前位姿（get_pose）
          2. 计算 6D 误差（position_error + orientation_error）
          3. 读取雅可比
          4. IKSolver.compute_joint_delta → Δθ
          5. 应用步长增益 + 关节限位裁剪
          6. 下发 PD 目标
          7. 默认保持夹爪打开（open_gripper=True）；抓取后抬升时传 False 以保持夹紧

        Args:
            open_gripper: 是否在每步后张开夹爪（抓取后应传 False 以免掉落物体）

        Returns:
            (position_error_norm, rotation_error_norm)
        """
        current_position, current_orientation = self.get_pose()
        position_error = target_position - current_position
        rotation_error = orientation_error(target_orientation, current_orientation)

        ee_jacobian = self._get_ee_jacobian()

        delta = self._ik.compute_joint_delta(
            ee_jacobian, position_error, rotation_error
        )

        # 应用步长增益和限幅
        command = self.get_arm_positions() + ik_config.step_gain * delta
        margin = math.radians(0.5)
        command = np.clip(
            command, self.lower_limits + margin, self.upper_limits - margin
        )
        self._robot.set_dof_position_targets(
            command[np.newaxis, :], dof_indices=self._robot.arm_indices
        )
        if open_gripper:
            self._gripper.command_open()

        return float(np.linalg.norm(position_error)), float(
            np.linalg.norm(rotation_error)
        )

    # =========================================================================
    # 笛卡尔空间路径规划（核心方法）
    # =========================================================================

    def follow_pose_path(
        self,
        goal_position: np.ndarray,
        goal_orientation: np.ndarray,
        path_steps: int,
        settle_steps: int,
        open_gripper: bool = True,
    ) -> tuple[float, float]:
        """沿笛卡尔直线运动到目标位姿。

        两阶段：
          1. path_steps: 使用 smoothstep5 插值位置 + slerp 插值姿态，
                       每步调用 command_pose() 跟踪插值路径。
          2. settle_steps: 保持在目标位姿，持续 IK 闭环直到误差小于容差
                        或步数耗尽。

        Args:
            goal_position:     目标位置 (3,)
            goal_orientation:  目标姿态四元数 (4,)
            path_steps:        插值步数（越多越平滑）
            settle_steps:      收敛步数
            open_gripper:      是否在每步后张开夹爪（抓取后传 False）

        Returns:
            (最终位置误差模长, 最终角度误差模长)
        """
        start_position, start_orientation = self.get_pose()

        # ---- 第一阶段：路径插值 -------------------------------------------
        for step in range(path_steps):
            # smoothstep5 保证起终点速度和加速度均为零
            phase = smoothstep5((step + 1) / path_steps)

            # 位置：线性插值
            waypoint_position = start_position + phase * (
                goal_position - start_position
            )
            # 姿态：球面线性插值
            waypoint_orientation = quaternion_slerp(
                start_orientation, goal_orientation, phase
            )
            self.command_pose(waypoint_position, waypoint_orientation, open_gripper=open_gripper)
            self._sim.update()

        # ---- 第二阶段：收敛 -----------------------------------------------
        for _ in range(settle_steps):
            position_error, orientation_error_value = self.command_pose(
                goal_position, goal_orientation, open_gripper=open_gripper
            )
            self._sim.update()
            if (
                position_error < grasp_config.position_tolerance
                and orientation_error_value < grasp_config.orientation_tolerance
            ):
                break

        # ---- 返回最终误差 ------------------------------------------------
        final_position, final_orientation = self.get_pose()
        return (
            float(np.linalg.norm(goal_position - final_position)),
            float(
                np.linalg.norm(
                    orientation_error(goal_orientation, final_orientation)
                )
            ),
        )

    # =========================================================================
    # 夹爪控制（委托给 GripperController）
    # =========================================================================

    def actuate_gripper(self, target: float, steps: int, contact_sensor: ContactSensor|None=None) -> None:
        """在 steps 步内驱动夹爪到 target 位置，同时保持臂关节不动。
        
        Args:
            target: 目标位置（0.0 = 张开，0.8203 = 闭合）
            steps:  驱动步数
        """
        arm_hold = self.get_arm_positions()
        self._gripper.actuate(target, steps, arm_hold, contact_sensor)

    # =========================================================================
    # 关节空间路径规划（备用方法）
    # =========================================================================

    def follow_joint_path(
        self,
        goal_joints: np.ndarray,
        goal_position: np.ndarray,
        goal_orientation: np.ndarray,
        steps: int,
        open_gripper: bool = True,
    ) -> tuple[float, float]:
        """沿关节空间直线运动到目标关节角。

        用于不需要笛卡尔直线精度的场合（如避障路径）。
        实际步数 = max(steps, max_delta / max_joint_step)，保证单步变化不过大。

        Args:
            open_gripper: 是否在每步后张开夹爪（抓取后传 False）

        Returns:
            (最终位置误差, 最终角度误差)
        """
        start = self.get_arm_positions()
        max_delta = float(np.max(np.abs(goal_joints - start)))
        effective_steps = max(
            steps, int(math.ceil(max_delta / grasp_config.max_joint_step))
        )
        for step in range(effective_steps):
            phase = smoothstep5((step + 1) / effective_steps)
            command = start + phase * (goal_joints - start)
            self._robot.set_dof_position_targets(
                command[np.newaxis, :], dof_indices=self._robot.arm_indices
            )
            if open_gripper:
                self._gripper.command_open()
            self._sim.update()
        # 保持 180 步让物理稳定
        for _ in range(180):
            self._robot.set_dof_position_targets(
                goal_joints[np.newaxis, :], dof_indices=self._robot.arm_indices
            )
            if open_gripper:
                self._gripper.command_open()
            self._sim.update()

        actual_position, actual_orientation = self.get_pose()
        return (
            float(np.linalg.norm(goal_position - actual_position)),
            float(
                np.linalg.norm(
                    orientation_error(goal_orientation, actual_orientation)
                )
            ),
        )

    # =========================================================================
    # Home 位姿管理
    # =========================================================================

    def set_home_pose_now(self) -> None:
        """瞬间设置机械臂到 home 位姿（关节瞬间跳转）。
        
        步骤：
          1. 计算 home_joints（来自 config），裁剪到限位内
          2. 用 set_dof_positions 直接跳转（跳过物理驱动）
          3. 清零关节速度避免抖动
          4. 保持 20 帧让物理稳定
        """
        home = np.clip(
            arm_config.home_joints,
            self.lower_limits + math.radians(0.5),
            self.upper_limits - math.radians(0.5),
        )
        full_state = self._robot.get_dof_positions()
        full_state[: arm_config.arm_dof_count] = home
        full_state[self._gripper.gripper_index] = gripper_config.open_position
        self._robot.set_dof_positions(full_state[np.newaxis, :])
        # 尝试清零速度（非所有 Articulation 都支持）
        try:
            self._robot.set_dof_velocities(
                np.zeros((1, len(self._robot.dof_names)), dtype=np.float64)
            )
        except Exception as exc:
            print(f"[WARN] 清零关节速度失败，继续运行：{exc}")
        self._robot.set_dof_position_targets(
            home[np.newaxis, :], dof_indices=self._robot.arm_indices
        )
        self._gripper.command_open()
        for _ in range(20):
            self._sim.update()
        print(
            f"[INFO] 初始姿态已稳定设为 /arm 高位弯肘 home：{np.round(home, 5).tolist()}"
        )

    def return_home(self, steps: int) -> None:
        """平滑回到 home 位姿（关节空间 smoothstep5 插值）。
        
        Args:
            steps: 插值步数
        """
        start = self.get_arm_positions()
        home = np.clip(
            arm_config.home_joints,
            self.lower_limits + math.radians(0.5),
            self.upper_limits - math.radians(0.5),
        )
        for step in range(steps):
            phase = smoothstep5((step + 1) / steps)
            command = start + phase * (home - start)
            self._robot.set_dof_position_targets(
                command[np.newaxis, :], dof_indices=self._robot.arm_indices
            )
            self._gripper.command_open()
            self._sim.update()
        # 保持 60 步稳定
        for _ in range(60):
            self._robot.set_dof_position_targets(
                home[np.newaxis, :], dof_indices=self._robot.arm_indices
            )
            self._gripper.command_open()
            self._sim.update()
