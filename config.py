# =============================================================================
# config.py — 全局配置常量
#
# 所有可调参数集中在此处管理，业务层通过 import 使用。
# 纯 Python/numpy 类型，不依赖 isaacsim，可在 SimulationApp() 前安全导入。
# =============================================================================

from dataclasses import dataclass, field
import math

import numpy as np


# =============================================================================
# AppConfig — 仿真应用基本设置
# =============================================================================
@dataclass
class AppConfig:
    # USD 场景文件路径 — 包含机械臂、方块、工作台的完整仿真场景
    usd_path: str = "data/1.usd"

    # USD 场景中的关节链路径
    # 场景结构：/arm/gripper/Robotiq_2F_85（一个 Articulation 包含臂 + 夹爪）
    articulation_path: str = "/arm/gripper/Robotiq_2F_85"

    headless: bool = False      # 无窗口模式（适合 CI/远程）
    width: int = 1280           # 渲染窗口宽度
    height: int = 720           # 渲染窗口高度
    physics_dt: float = 1.0 / 60.0  # 物理时间步长（60 Hz）


# =============================================================================
# 初始 home 位姿（7 个关节角度，单位：度）
# 在 USD 编辑器中手动调试确定的"高位弯肘"姿态，
# 避免机械臂与工作台或其他物体碰撞。
# =============================================================================
TARGET_JOINT_POS_DEG = [
    -30.0,    # joint_1（基座旋转）
    30.0,     # joint_2（肩部）
    0.0,      # joint_3（上臂）
    70.0,     # joint_4（肘部）
    0.0,      # joint_5（前臂旋转）
    70.0,     # joint_6（腕部俯仰）
    0.0,      # joint_7（腕部旋转）
]


# =============================================================================
# ArmConfig — 机械臂结构参数
# =============================================================================
@dataclass
class ArmConfig:
    # 夹爪左右手指的 link 名称（用于计算手指中点 TCP）
    left_finger_link: str = "left_inner_finger"
    right_finger_link: str = "right_inner_finger"

    # 夹爪 DOF 名称（Articulation 中控制开合的关节）
    gripper_dof_name: str = "finger_joint"

    # 机械臂关节数量（前 N 个 DOF 属于臂，其余属于夹爪）
    arm_dof_count: int = 7

    # IK 控制点 — link7（法兰盘末端），不再使用手指中点
    end_effector_link: str = "AR5_5_08L_W4C1C5_link7"

    # home 位姿（弧度），从 TARGET_JOINT_POS_DEG 转换而来
    home_joints: np.ndarray = field(
        default_factory=lambda: np.radians(
            np.array(TARGET_JOINT_POS_DEG, dtype=np.float64)
        )
    )


# =============================================================================
# GripperConfig — 夹爪开合位置
# =============================================================================
@dataclass
class GripperConfig:
    open_position: float = 0.0        # 全开位置（弧度）
    closed_position: float = 0.82030475  # 全闭位置（弧度）


# =============================================================================
# GraspConfig — 抓取规划参数
#
# 用于 TargetGenerator 和 PoseIKPlanner 中的碰撞检测、容差判断。
# =============================================================================
@dataclass
class GraspConfig:
    target_width: float = 0.03                      # 目标物体宽度
    target_center_z: float = 0.03 / 2.0             # 目标中心高度偏移
    grasp_min_z: float = 0.055                      # 抓取最小高度
    grasp_max_z: float = 0.18                       # 抓取最大高度
    ground_clearance: float = 0.030                 # 离地间隙
    final_grasp_ee_z: float = 0.03 + 0.008          # 最终抓取高度
    final_descend_max_distance: float = 0.075        # 单次下降最大距离
    top_down_max_angle: float = math.radians(18.0)   # 顶抓最大偏角
    max_joint_step: float = math.radians(0.35)       # 单步最大关节变化
    max_target_joint_delta: float = math.radians(90.0)  # 最大目标关节变化
    axis_length: float = 0.065                       # 可视化辅助轴长度
    axis_radius: float = 0.0015                      # 可视化辅助轴半径
    position_tolerance: float = 0.025                # 位置收敛容差（米）
    orientation_tolerance: float = math.radians(7.0)  # 姿态收敛容差（弧度）


# =============================================================================
# TrajectoryConfig — 常规轨迹步数
# =============================================================================
@dataclass
class TrajectoryConfig:
    path_steps: int = 240       # 路径插值步数（越大运动越平滑）
    settle_steps: int = 360     # 到位后收敛步数
    gripper_steps: int = 240    # 夹爪动作步数
    home_steps: int = 240       # 回到 home 的步数


# =============================================================================
# IKConfig — 差分 IK 求解器参数
# =============================================================================
@dataclass
class IKConfig:
    damping: float = 0.10                # 阻尼系数（越大越保守）
    max_delta_per_step: float = 0.022    # 单步最大关节变化
    step_gain: float = 0.50              # IK 输出步长增益
    rotation_weight: float = 0.50        # 旋转误差权重（未使用，保留）


# =============================================================================
# TestConfig — 快速测试模式的短步数
# 通过 --test 命令行参数切换，用于调试时快速验证逻辑正确性。
# =============================================================================
@dataclass
class TestConfig:
    path_steps: int = 100       # 路径步数（约为常规的 40%）
    settle_steps: int = 120     # 收敛步数
    gripper_steps: int = 70     # 夹爪动作步数
    home_steps: int = 100       # 回 home 步数


# =============================================================================
# 单例实例 — 模块级常量，其他文件直接导入使用
# =============================================================================
app_config = AppConfig()

arm_config = ArmConfig()
gripper_config = GripperConfig()
grasp_config = GraspConfig()
trajectory_config = TrajectoryConfig()
ik_config = IKConfig()
test_config = TestConfig()
