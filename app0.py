# =============================================================================
# app0.py — 3 点顺序抓取演示程序
#
# 功能：控制 Rokae 机械臂 + Robotiq 2F-85 夹爪，依次到达 3 个预抓取位置，
#       在每个位置执行：下降到抓取高度 → 闭合/打开夹爪 → 抬升。
#
# 依赖模块（同目录下）：
#   config.py          — 所有配置常量（场景路径、步数、关节限位等）
#   math_utils.py      — 四元数运算、RPY 转换、光滑插值等数学工具
#   robot.py           — RobotArm（机械臂 Articulation 封装）和 GripperController（夹爪控制）
#   sim_app.py         — SimulationManager（SimulationApp 的 Singleton 封装）
#   ik_solver.py       — 差分 IK 求解器（SVD / 伪逆 / 阻尼最小二乘）
#   pose_planner.py    — PoseIKPlanner（IK 路径规划器，整合了 IK、雅可比、轨迹插值）
#
# 调用入口：python -m A-rokae-app （或 python -m A-rokae-app app0）
#           python -m A-rokae-app app1  → 切换到文件监听模式（编辑 YAML 驱动）
# =============================================================================

import argparse
import math
import os
import sys

import numpy as np
from isaacsim import SimulationApp

# ---- 路径设置：把项目根目录加入 sys.path，使得 .config 等包导入可用 ----
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# ---- 命令行参数 ----
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="无窗口运行")
parser.add_argument("--test", action="store_true", help="缩短轨迹步数（调试用）")
args, _ = parser.parse_known_args()

# ---- 从 config.py 导入配置常量 ----
# 这些配置在 SimulationApp() 之前导入，因为只含纯 Python/numpy 数据，
# 不触发 isaacsim 的底层初始化。
from .config import (
    app_config,       # AppConfig：USD 路径、渲染尺寸、物理 dt
    gripper_config,   # GripperConfig：夹爪开/合位置
    test_config as tc,       # TestConfig：测试模式的短步数
    trajectory_config as trc,  # TrajectoryConfig：常规步数
)

# ---- 创建 Isaac Sim 仿真应用 ----
# 必须在任何 isaacsim.core / pxr 导入之前调用，
# 否则 Isaac Sim 的底层模块尚未初始化会报错。
simulation_app = SimulationApp(
    {
        "headless": args.headless,
        "width": app_config.width,
        "height": app_config.height,
        "open_usd": app_config.usd_path,
    }
)

# ---- 现在可以安全导入 isaacsim 下的业务模块 ----
from .math_utils import quaternion_to_rpy, rpy_to_quaternion
#   rpy_to_quaternion(roll, pitch, yaw) → [w,x,y,z]  用于构建目标姿态
#   quaternion_to_rpy([w,x,y,z]) → (roll, pitch, yaw) 用于打印当前姿态

from .pose_planner import PoseIKPlanner
#   PoseIKPlanner 是核心 IK 规划器，整合了：
#   - IKSolver（阻尼最小二乘逆运动学）
#   - 末端执行器雅可比（link7 的 6x7 雅可比矩阵）
#   - 轨迹插值（quintic smoothstep + 四元数 slerp）

from .robot import GripperController, RobotArm
#   RobotArm：对 Isaac Sim Articulation 的封装，提供 get_arm_positions()、set_dof_position_targets() 等
#   GripperController：封装夹爪 DOF，提供 actuate() 方法开合夹爪

from .sim_app import SimulationManager
#   SimulationManager：Singleton 封装，持有 SimulationApp 引用，

from isaacsim.sensors.experimental.physics import Contact, ContactSensor, EffortSensor
#   ContactSensor：读取 Contact_Sensor prim 的接触力 / 碰撞状态

# ---- 检查 USD 场景文件是否存在 ----
if not os.path.isfile(app_config.usd_path):
    raise FileNotFoundError(f"找不到 USD 场景：{app_config.usd_path}")

import isaacsim.core.experimental.utils.app as app_utils
#   app_utils.play() 让仿真从 "暂停" 进入 "播放" 状态
#   （必须在 SimulationApp() 创建之后导入）

# =============================================================================
# 预抓取位置数据
#
# 这些坐标从 USD 场景中方块（/Obs/RedCube 等）的包围盒中心测量得到。
# 预抓取高度 z=0.4m，姿态 roll=180°（夹爪朝下），然后垂直下降到 z=0.25m 执行抓取。
# =============================================================================

PREGRASP_POSES = (
    {
        "label": "Blue",
        "x": 0.05055268853902817,
        "y": -0.35494229197502136,
        "z": 0.4,
        "roll": 180.0,
        "pitch": 0.0,
        "yaw": 0.0,
    },
    {
        "label": "Yellow",
        "x": 0.26470720767974854,
        "y": -0.1874564290046692,
        "z": 0.4,
        "roll": 180.0,
        "pitch": 0.0,
        "yaw": 0.0,
    },
    {
        "label": "Red",
        "x": 0.38922354578971863,
        "y": 0.01155999954789877,
        "z": 0.4,
        "roll": 180.0,
        "pitch": 0.0,
        "yaw": 0.0,
    },
)

DESCEND_Z = 0.25  # 夹爪下降到该高度后执行闭合/打开


def print_status(
    planner: PoseIKPlanner,
    label: str = "",
    pos_err: float | None = None,
    rot_err: float | None = None,
) -> None:
    """打印机械臂当前状态摘要：关节角 + 末端 TCP 位姿 + 路径规划误差。

    三项内容在 PoseIKPlanner 中的来源：
    - 关节角：  get_arm_positions()        → 7 个关节弧度值，转度打印
    - TCP 位姿：get_pose()                  → link7 的世界坐标 (m) + 四元数 → RPY (deg)
    - 规划误差：follow_pose_path 的返回值  → 到达目标后的位置/角度残余误差
    TCP 是 link7 的坐标系原点（即机械臂法兰盘中心），不再计算夹爪手指中点。
    """
    joints = planner.get_arm_positions()
    pos, ori = planner.get_pose()
    rpy = quaternion_to_rpy(ori)
    parts = [
        f"[状态{label}]",
        f"关节(deg)={np.round(np.degrees(joints), 2).tolist()}\n",
        f"TCP=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})",
        f"RPY(deg)=({math.degrees(rpy[0]):.1f}, {math.degrees(rpy[1]):.1f}, {math.degrees(rpy[2]):.1f})",
    ]
    if pos_err is not None and rot_err is not None:
        parts.append(f"err_pos={pos_err:.4f}m err_rot={math.degrees(rot_err):.2f}deg")
    print(" | ".join(parts))


def main() -> None:
    # =========================================================================
    # 第一步：初始化仿真环境
    # =========================================================================

    sim_mgr = SimulationManager(simulation_app)
    # setup_simulation(dt) 调用 Isaac Sim 的 SimulationManager.setup_simulation()
    # 设置物理步长，启动物理引擎。
    sim_mgr.setup_simulation(dt=app_config.physics_dt)

    # 闲置步进 100 帧，让物理引擎稳定、场景完全加载
    for _ in range(100):
        sim_mgr.update()

    app_utils.play()  # 切换到 "播放" 模式（物理模拟开始推进）
    for _ in range(60):
        sim_mgr.update()

    # =========================================================================
    # 第二步：初始化机器人 + IK 规划器
    # =========================================================================

    # RobotArm 封装了 Articulation("/arm/gripper/Robotiq_2F_85")，
    # 并从中提取 arm_indices=[0..6]（前 7 个 DOF 为机械臂关节）和 finger RigidPrim。
    robot = RobotArm(app_config.articulation_path)

    # GripperController 找到 "finger_joint" 及其关联 DOF，
    # 提供 actuate(target, steps) 进行开合控制。
    gripper = GripperController(robot)

    # PoseIKPlanner 是核心：
    # - 根据 config.end_effector_link 找到 link7 作为控制点
    # - 创建 IKSolver（damped-least-squares 方法）
    # - 读取关节限位用于安全裁剪
    planner = PoseIKPlanner(robot, gripper)

    # set_home_pose_now()：将 7 个关节瞬间设到 home 角度，
    # 然后保持 20 帧让物理引擎稳定。
    planner.set_home_pose_now()
    # planner.actuate_gripper(gripper_config.open_position, 90)
    print_status(planner, " 初始")

    # 创建接触传感器，用于检测夹爪与物体是否接触
    _contact = Contact.create(
        "/arm/gripper/Robotiq_2F_85/right_inner_finger/Contact_Sensor",
        min_threshold=1.0,
        max_threshold=1000.0,
        radius=0.05,
    )

    contact_sensor = ContactSensor(_contact) # type: ignore
    
    # Get current contact data
    # "in_contact": Whether contact is detected.
    # "force": Contact force magnitude.
    # "time": Simulation time of reading.
    # "physics_step": Physics step number.
    # "number_of_contacts": Number of contact points.
    # "contacts": Raw contact data if enabled via add_raw_contact_data_to_frame.
    frame = contact_sensor.get_data()

    # 创建 Effort 传感器，测量 finger_joint（夹爪主驱动关节）的力矩
    # 注：right_inner_finger_joint 是耦合被动关节，PhysX 不单独计算其力矩值，
    # 因此改用主驱动关节 finger_joint。
    effort_sensor = EffortSensor(path="/arm/gripper/Robotiq_2F_85/finger_joint")

    # 将 sleepThreshold 设为 0 使刚体永不休眠，接触数据持续可用。
    # from pxr import PhysxSchema
    # import isaacsim.core.experimental.utils.stage as stage_utils
    # stage = stage_utils.get_current_stage()
    # ant_root_prim = stage.GetPrimAtPath("/arm/gripper/Robotiq_2F_85/right_inner_finger") # type: ignore
    # if not ant_root_prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
    #     PhysxSchema.PhysxArticulationAPI.Apply(ant_root_prim)
    # art_api = PhysxSchema.PhysxArticulationAPI(ant_root_prim)
    # art_api.GetSleepThresholdAttr().Set(0.0)

    # =========================================================================
    # 第三步：选择轨迹步数（普通/测试模式）
    # =========================================================================

    path_steps = tc.path_steps if args.test else trc.path_steps
    #   path_steps：   路径插值步数，值越大运动越平滑（但越慢）
    #   settle_steps： 到达目标后的收敛步数，用于让 IK 误差收敛到 tolerance 以内
    #   gripper_steps：夹爪开合动作的步数
    settle_steps = tc.settle_steps if args.test else trc.settle_steps
    gripper_steps = tc.gripper_steps if args.test else trc.gripper_steps

    # =========================================================================
    # 第四步：遍历 3 个预抓取位置执行动作序列
    # =========================================================================

    for pose in PREGRASP_POSES:
        label = pose["label"]
        # 构建目标位姿
        target_pos = np.array([pose["x"], pose["y"], pose["z"]], dtype=np.float64)
        rpy_deg = (pose["roll"], pose["pitch"], pose["yaw"])
        target_ori = rpy_to_quaternion(
            math.radians(pose["roll"]),
            math.radians(pose["pitch"]),
            math.radians(pose["yaw"]),
        )

        print(f"\n[INFO] === {label} ===")
        print(f"[INFO] 预抓取位姿 pos=({pose['x']:.4f}, {pose['y']:.4f}, {pose['z']}) rpy={rpy_deg}")

        # --- 4a: 从当前位置路径规划到预抓取位姿 ---
        # follow_pose_path(goal_pos, goal_ori, path_steps, settle_steps)
        # 内部流程：
        #   1. 记录起始位姿
        #   2. path_steps 步中，每步用 smoothstep5(t) 线性插值位置，
        #      用 quaternion_slerp 插值姿态
        #   3. 每步调用 command_pose() → 计算雅可比 → IK 解关节增量 → 下发 → 步进仿真
        #   4. settle_steps 步在目标位姿上持续修正，直到 position/orientation 误差
        #      小于 grasp_config 中的 tolerance 或步数耗尽
        planner.follow_pose_path(target_pos, target_ori, path_steps, settle_steps)
        print_status(planner, f" {label} pre")

        # --- 4b: 垂直下降到抓取高度 ---
        descend_pos = target_pos.copy()
        descend_pos[2] = DESCEND_Z
        print(f"[INFO] 下降到 z={DESCEND_Z}")
        planner.follow_pose_path(descend_pos, target_ori, path_steps, settle_steps)
        print_status(planner, f" {label} down")

        # --- 4c: 夹爪闭合（抓取方块） ---
        # actuate_gripper(target, steps) 调用 GripperController.actuate()
        # 在 steps 步内将夹爪 DOF 从当前位置平滑驱动到 target，
        # 同时保持机械臂关节不动（用于 hold 的 arm_hold 值）。
        # gripper_config.closed_position ≈ 0.8203（弧度，对应夹紧状态）
        print(f"[INFO] 闭合夹爪")
        planner.actuate_gripper(gripper_config.closed_position, gripper_steps, contact_sensor)

        print(f"[INFO] 夹爪力矩={effort_sensor.get_data()}")
        print(f"[INFO] 接触传感器数据={contact_sensor.get_data()}")

        # --- 4d: 夹住方块抬升 ---
        # open_gripper=False → command_pose 不会再每步调用 command_open()，
        # 保持夹爪夹紧，方块不会掉落。
        print(f"[INFO] 夹住方块抬升到 z={pose['z']}")
        planner.follow_pose_path(target_pos, target_ori, path_steps, settle_steps, open_gripper=False)
        print_status(planner, f" {label} up")

        # --- 4e: 夹住方块放回原处（下降到 DESCEND_Z）---
        print(f"[INFO] 夹住方块下降到抓取高度 z={DESCEND_Z}")
        planner.follow_pose_path(descend_pos, target_ori, path_steps, settle_steps, open_gripper=False)
        print_status(planner, f" {label} down_place")

        # --- 4f: 打开夹爪（释放方块）---
        print(f"[INFO] 打开夹爪，释放方块")
        planner.actuate_gripper(gripper_config.open_position, gripper_steps, contact_sensor)

        # --- 4g: 空夹爪抬升回预抓取高度 ---
        print(f"[INFO] 空夹爪抬升到 z={pose['z']}")
        planner.follow_pose_path(target_pos, target_ori, path_steps, settle_steps)
        print_status(planner, f" {label} up_empty")

    # =========================================================================
    # 第五步：回到 home 位姿
    # =========================================================================

    print("[INFO] 序列结束，回到 home")
    home_steps = tc.home_steps if args.test else trc.home_steps
    # return_home(steps)：用 smoothstep5 从当前位置平滑回到 home 关节角度
    planner.return_home(home_steps)
    print_status(planner, " home")

    # 有窗口时保持打开直到用户关闭
    if not args.headless:
        print("[INFO] 关闭窗口退出")
        while simulation_app.is_running():
            sim_mgr.update()


def app_main():
    """__main__.py 入口函数。打包为模块入口时的统一调用接口。"""
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    finally:
        simulation_app.close()
