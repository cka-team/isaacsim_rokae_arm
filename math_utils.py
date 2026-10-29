# =============================================================================
# math_utils.py — 数学工具函数
#
# 提供机械臂控制所需的数学运算：
#   四元数运算  → 姿态表示与插值
#   RPY 转换    → 人可读的角度表示
#   差分 IK 相关 → 旋转误差、skew 矩阵
#   轨迹平滑    → smoothstep5 五次光滑插值
#   数值辅助    → to_numpy、set_prim_visibility
# =============================================================================

import math

import numpy as np


# ---------------------------------------------------------------------------
# 数值转换辅助
# ---------------------------------------------------------------------------
def to_numpy(value) -> np.ndarray:
    """统一将 Isaac Sim 的 wp.array 或 Python 数值转为 np.ndarray。"""
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


# ---------------------------------------------------------------------------
# 五次光滑 step 函数（quintic smoothstep）
#
# 用于轨迹插值，保证位置和速度在起点和终点均为零（C2 连续）。
# 相比三次 smoothstep，它能提供连续的加速度，减少机械臂运动冲击。
#
# 公式：6t^5 - 15t^4 + 10t^3
#       在 t=0 和 t=1 处，一阶和二阶导数均为零。
# ---------------------------------------------------------------------------
def smoothstep5(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5


# ---------------------------------------------------------------------------
# 四元数标准化
# ---------------------------------------------------------------------------
def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    """将四元数归一化为单位四元数。零向量回退到单位四元数 [1,0,0,0]。
    
    四元数格式：[w, x, y, z]
    """
    norm = np.linalg.norm(quaternion)
    if norm < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quaternion / norm


def quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    """四元数共轭：实部不变，虚部取反。
    
    对于单位四元数，共轭等于逆（用于旋转求逆）。
    """
    w, x, y, z = quaternion
    return np.array([w, -x, -y, -z], dtype=np.float64)


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """四元数乘法（Hamilton 积）。
    
    两个旋转的合成：先右后左。
    """
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# RPY ↔ 四元数 转换
# ---------------------------------------------------------------------------
def rpy_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """欧拉角（Roll-Pitch-Yaw）→ 四元数 [w, x, y, z]。
    
    转换顺序：ZYX（先 Yaw 绕 Z，再 Pitch 绕 Y，最后 Roll 绕 X）。
    这就是 app0.py 中 PREGRASP_POSES 里 rpy_deg → target_ori 的转换。
    """
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    return normalize_quaternion(
        np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=np.float64,
        )
    )


def quaternion_to_rpy(quaternion: np.ndarray) -> np.ndarray:
    """四元数 → 欧拉角（弧度），返回 [roll, pitch, yaw]。
    
    用于 print_status() 中打印当前姿态。
    注意：在 gimbal lock 附近 pitch=±90° 时，roll 和 yaw 可能不唯一。
    """
    w, x, y, z = normalize_quaternion(quaternion)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_value = 2.0 * (w * y - z * x)
    pitch = math.asin(min(max(pitch_value, -1.0), 1.0))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float64)


# ---------------------------------------------------------------------------
# 旋转误差
#
# 用于 IK 闭环中的姿态误差反馈。
# 计算从当前姿态到目标姿态的最小旋转轴×角度。
# 返回三维向量，方向 = 旋转轴，模长 = 旋转角（弧度）。
# ---------------------------------------------------------------------------
def orientation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """计算目标四元数与当前四元数之间的旋转误差向量。
    
    公式：error = 2 * log(target * conjugate(current))
    对单位四元数简化为：取四元数差值的虚部 × 2*arctan(|v|, w) 比例。
    
    返回三维向量，可直接用于差分 IK 的旋转误差项。
    """
    error = quaternion_multiply(
        normalize_quaternion(target), quaternion_conjugate(normalize_quaternion(current))
    )
    if error[0] < 0.0:
        error = -error
    vector = error[1:]
    vector_norm = np.linalg.norm(vector)
    if vector_norm < 1e-10:
        return 2.0 * vector
    angle = 2.0 * math.atan2(vector_norm, max(error[0], 1e-10))
    return vector / vector_norm * angle


# ---------------------------------------------------------------------------
# 四元数旋转向量
# ---------------------------------------------------------------------------
def quaternion_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """用四元数旋转三维向量：将 vector 视为纯四元数，执行 q * v * q⁻¹。
    
    用于将关节空间量转换到世界坐标系。
    """
    quaternion = normalize_quaternion(quaternion)
    pure = np.array([0.0, vector[0], vector[1], vector[2]], dtype=np.float64)
    return quaternion_multiply(
        quaternion_multiply(quaternion, pure), quaternion_conjugate(quaternion)
    )[1:]


# ---------------------------------------------------------------------------
# 叉积的矩阵形式：skew(v) * w = v × w
# ---------------------------------------------------------------------------
def skew(vector: np.ndarray) -> np.ndarray:
    """构造三维向量的斜对称矩阵（so(3)）。
    
    用途：将叉积运算表示为矩阵乘法。在机器人学中常用于：
    - 几何雅可比计算
    - 旋转矩阵的指数映射
    """
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


# ---------------------------------------------------------------------------
# 四元数球面线性插值（Spherical Linear Interpolation）
#
# 在起始和终止四元数之间沿球面大圆插值，保持匀速旋转。
# 用于 follow_pose_path 中每步的姿态插值。
#
# 特殊处理：
#   - dot < 0 时取反（走短弧）
#   - dot > 0.9995 时退化为线性插值（避免除零）
# ---------------------------------------------------------------------------
def quaternion_slerp(start: np.ndarray, goal: np.ndarray, t: float) -> np.ndarray:
    start = normalize_quaternion(start)
    goal = normalize_quaternion(goal)
    dot = float(np.dot(start, goal))
    if dot < 0.0:
        goal = -goal
        dot = -dot
    dot = min(max(dot, -1.0), 1.0)
    if dot > 0.9995:
        return normalize_quaternion(start + t * (goal - start))
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    return normalize_quaternion(
        math.sin((1.0 - t) * theta) / sin_theta * start
        + math.sin(t * theta) / sin_theta * goal
    )


def set_prim_visibility(stage, prim_path: str, visible: bool) -> None:
    """设置 USD Prim 的可见性（显示/隐藏），用于场景中物体的显示控制。"""
    from pxr import UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    imageable = UsdGeom.Imageable(prim)
    if visible:
        imageable.MakeVisible()
    else:
        imageable.MakeInvisible()
