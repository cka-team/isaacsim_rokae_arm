# =============================================================================
# robot.py — 机械臂（RobotArm）与夹爪控制器（GripperController）
#
# 封装 Isaac Sim 的 Articulation API，提供面向应用程序的接口：
#   RobotArm           → 关节位置读写、雅可比获取、关节限位
#   GripperController  → 夹爪开合驱动（含保持逻辑）
# =============================================================================

import math

from isaacsim.sensors.experimental.physics import ContactSensor
import numpy as np
from isaacsim.core.experimental.prims import Articulation, RigidPrim

from .config import arm_config, gripper_config
from .math_utils import to_numpy
from .sim_app import SimulationManager


# =============================================================================
# RobotArm — 机械臂总成
#
# 封装 Articulation("/arm/gripper/Robotiq_2F_85")，内部包含：
#   - 7 个臂关节（DOF 0-6，对应 joint_1 ~ joint_7）
#   - 若干个夹爪 DOF（finger_joint、knuckle_joints 等）
#   - left_inner_finger / right_inner_finger 两个 link（用于初始 TCP）
#
# 职责：
#   - 读写关节位置（get_dof_positions / set_dof_position_targets）
#   - 获取关节限位（用于安全裁剪）
#   - 获取雅可比矩阵（用于 IK 求解）
#   - 记录左右手指 link 索引（前期版本用于中点 TCP，当前版本 TCP 是 link7）
# =============================================================================
class RobotArm:
    def __init__(self, articulation_path: str):
        # Articulation 是 Isaac Sim 中操作关节链的核心 API。
        # 它关联 USD 场景中指定路径下的所有关节和 link。
        self._articulation = Articulation(articulation_path)

        # 前 7 个 DOF 是机械臂关节（joint_1 ~ joint_7）
        self.arm_indices = list(range(arm_config.arm_dof_count))

        # 验证左右手指 link 在 Articulation 中存在
        for link_name in (arm_config.left_finger_link, arm_config.right_finger_link):
            if link_name not in self._articulation.link_names:
                raise RuntimeError(
                    f"找不到 {link_name}；可用 links：{self._articulation.link_names}"
                )

        # 创建手指 RigidPrim（用于上一版本的 TCP 计算，当前保留用于 GripperController）
        left_index = self._articulation.link_names.index(arm_config.left_finger_link)
        right_index = self._articulation.link_names.index(arm_config.right_finger_link)
        self._left_finger = RigidPrim(self._articulation.link_paths[0][left_index])
        self._right_finger = RigidPrim(self._articulation.link_paths[0][right_index])

    # ---- 属性暴露（委托给底层 Articulation） --------------------------------

    @property
    def articulation(self):
        return self._articulation

    @property
    def dof_names(self):
        """所有 DOF 的名称列表（包括臂 DOF 和夹爪 DOF）。"""
        return self._articulation.dof_names

    @property
    def link_names(self):
        return self._articulation.link_names

    @property
    def link_paths(self):
        return self._articulation.link_paths

    @property
    def prims(self):
        return self._articulation.prims

    @property
    def left_finger(self) -> RigidPrim:
        return self._left_finger

    @property
    def right_finger(self) -> RigidPrim:
        return self._right_finger

    # ---- 关节控制 -----------------------------------------------------------

    def get_dof_positions(self) -> np.ndarray:
        """返回所有 DOF 的当前位置（弧度），形状 (1, n_dof)。"""
        return to_numpy(self._articulation.get_dof_positions())[0].astype(np.float64)

    def get_arm_positions(self) -> np.ndarray:
        """只返回机械臂 7 个关节的位置（夹爪 DOF 除外）。"""
        return self.get_dof_positions()[: arm_config.arm_dof_count]

    def set_dof_positions(self, positions: np.ndarray) -> None:
        """直接设置 DOF 位置（瞬间跳转，用于 set_home_pose_now）。"""
        self._articulation.set_dof_positions(positions)

    def set_dof_position_targets(
        self, targets: np.ndarray, dof_indices=None
    ) -> None:
        """设置 DOF 位置目标（物理引擎通过 PD 控制器驱动到达）。
        
        Args:
            targets: 目标位置（弧度）
            dof_indices: 可选，只设置指定索引的 DOF
        """
        self._articulation.set_dof_position_targets(targets, dof_indices=dof_indices)

    # ---- 关节限位与雅可比 ---------------------------------------------------

    def get_dof_limits(self, dof_indices=None):
        """获取 DOF 的 [下限, 上限]，用于 IK 输出裁剪。"""
        return self._articulation.get_dof_limits(dof_indices=dof_indices)

    def get_jacobian_matrices(self):
        """获取所有 link 的雅可比矩阵。
        
        返回形状通常为 (1, n_links, 6, n_dof)，由 PoseIKPlanner 按 ee_jacobian_index 提取。
        """
        return self._articulation.get_jacobian_matrices()

    def set_dof_velocities(self, velocities: np.ndarray) -> None:
        """设置 DOF 速度（用于 set_home_pose_now 中清零速度）。"""
        if hasattr(self._articulation, "set_dof_velocities"):
            self._articulation.set_dof_velocities(velocities)

    def get_link_index(self, link_name: str) -> int:
        """按名称查找 link 的索引。"""
        return self._articulation.link_names.index(link_name)


# =============================================================================
# GripperController — 夹爪控制器
#
# 职责：
#   1. 在 Articulation 的 DOF 中找到所有夹爪相关 DOF
#   2. 确定每个 DOF 的正负方向（有些会随 finger_joint 反向运动）
#   3. 提供高级接口 actuate() — 在指定步数内驱动夹爪到目标位置
#   4. 驱动夹爪时保持机械臂关节不动（arm_hold 机制）
# =============================================================================
class GripperController:
    def __init__(self, robot: RobotArm):
        self._robot = robot
        dof_names = list(robot.dof_names)

        # finger_joint 在 DOF 列表中的索引（用于 set_home_pose_now）
        self.gripper_index = dof_names.index(arm_config.gripper_dof_name)

        # 所有夹爪相关 DOF 的索引
        # Robotiq 2F-85 的夹爪包含多个耦合关节：
        # finger_joint、left_outer_knuckle_joint、right_outer_knuckle_joint 等
        self.gripper_indices = [
            index
            for index, name in enumerate(dof_names)
            if ("finger" in name or "knuckle" in name)
        ]
        self.gripper_names = [
            dof_names[index] for index in self.gripper_indices
        ]

    def get_gripper_targets(self, target: float) -> tuple[list[int], np.ndarray]:
        """计算各夹爪 DOF 的目标位置。
        
        由于 Robotiq 2F-85 的机械耦合设计：
        - finger_joint、right_outer_knuckle_joint、right_inner_finger_joint → 正方向运动
        - left_outer_knuckle_joint 等含 "inner" 的关节 → 反方向运动
        """
        targets = []
        for name in self.gripper_names:
            if name in (
                arm_config.gripper_dof_name,
                "right_outer_knuckle_joint",
                "right_inner_finger_joint",
            ):
                value = target
            elif "inner" in name:
                value = -target
            else:
                value = target
            targets.append(value)
        return self.gripper_indices, np.asarray(targets, dtype=np.float64)

    def actuate(self, target: float, steps: int, arm_hold: np.ndarray, contact_sensor: ContactSensor|None=None) -> None:
        """在 steps 步内将夹爪驱动到 target 位置，同时保持臂关节不动。
        
        流程：
          1. 用 arm_hold 固定所有臂关节的目标位置
          2. 每步设置夹爪 DOF 目标到 target
          3. 所有 DOF 目标同时下发 → sim.update() 推进物理
          4. 结束后额外 hold 若干步：
             - 如果 target > 0（闭合），hold 约 max(30, steps//3) 步让夹紧力稳定
             - 如果 target ≈ 0（打开），hold 10 步足够
        """
        sim = SimulationManager.instance()
        indices, targets = self.get_gripper_targets(target)
        
        # print(contact_sensor.get_data())

        for _ in range(steps):
            self._robot.set_dof_position_targets(
                arm_hold[np.newaxis, :], dof_indices=self._robot.arm_indices
            )
            self._robot.set_dof_position_targets(
                targets[np.newaxis, :], dof_indices=indices
            )
            # print(contact_sensor.get_data())
            sim.update()

        hold_steps = (
            max(30, steps // 3) if target > gripper_config.open_position + 1e-4 else 10
        )
        for _ in range(hold_steps):
            self._robot.set_dof_position_targets(
                arm_hold[np.newaxis, :], dof_indices=self._robot.arm_indices
            )
            self._robot.set_dof_position_targets(
                targets[np.newaxis, :], dof_indices=indices
            )
            sim.update()


    def command_open(self) -> None:
        """单步设置夹爪为张开状态。用于 command_pose() 中在每次 IK 步骤后保持夹爪打开。"""
        indices, targets = self.get_gripper_targets(gripper_config.open_position)
        self._robot.set_dof_position_targets(
            targets[np.newaxis, :], dof_indices=indices
        )
