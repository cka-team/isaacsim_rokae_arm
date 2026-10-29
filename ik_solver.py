# =============================================================================
# ik_solver.py — 差分逆运动学求解器
#
# 给定末端执行器的希望位姿变化（6D 误差：位置 + 旋转），
# 通过雅可比矩阵的伪逆/转置/阻尼最小二乘，求解对应的关节增量。
#
# 被 PoseIKPlanner.command_pose() 调用，位于 IK 闭环的最内层。
# =============================================================================

import math

import numpy as np
import warp as wp
from isaacsim.core.experimental.utils.impl.transform import (
    quaternion_conjugate,
    quaternion_multiplication,
)

from .math_utils import normalize_quaternion, orientation_error, to_numpy


# =============================================================================
# IKSolver — 差分 IK 求解器
#
# 数学原理：
#   对于冗余机械臂（7 DOF > 6 DOF 任务空间），雅可比矩阵 J 不是方阵。
#   我们需要解 Δθ = J⁺ · e，其中 J⁺ 是 J 的某种广义逆。
#
# 支持四种方法（通过 method 参数切换）：
#
#   1. singular-value-decomposition（SVD）
#      用 SVD 分解 J = U·Σ·Vᵀ，截断小奇异值后求伪逆。
#      最稳定但对奇异值截断敏感。
#
#   2. pseudoinverse
#      直接用 Moore-Penrose 伪逆 J⁺ = Jᵀ(J·Jᵀ)⁻¹。
#      在奇异位形附近会放大噪声。
#
#   3. transpose
#      直接用 Jᵀ 代替 J⁺，相当于梯度下降。
#      计算最快但收敛最慢。
#
#   4. damped-least-squares（默认，推荐）
#      Δθ = Jᵀ(J·Jᵀ + λ²I)⁻¹ · e
#      λ（damping）越大，解越平滑但收敛越慢。
#      在奇异位形附近仍能稳定工作。
#      本项目 PoseIKPlanner 传入 damping=0.05。
# =============================================================================
class IKSolver:
    def __init__(
        self,
        method: str = "damped-least-squares",
        damping: float = 0.05,
        scale: float = 1.0,
        min_singular_value: float = 1e-5,
    ):
        """
        Args:
            method: 求解方法，可选 "damped-least-squares" / "pseudoinverse" / "transpose" / "singular-value-decomposition"
            damping: 阻尼系数 λ（越大越平滑保守）
            scale: 输出缩放系数
            min_singular_value: SVD 截断阈值（仅用于 SVD 方法）
        """
        self.method = method
        self.damping = damping
        self.scale = scale
        self.min_singular_value = min_singular_value

    def compute_joint_delta(
        self,
        jacobian: np.ndarray,
        position_error: np.ndarray,
        rotation_error_vec: np.ndarray,
    ) -> np.ndarray:
        """计算关节增量 Δθ。

        核心方程：
          6D 误差 e = [position_error (3×1), rotation_error (3×1)]
          雅可比   J = [J_p; J_o]  (6×7)
          输出    Δθ = J⁺ · e      (7×1)

        Args:
            jacobian: 末端执行器雅可比 (6, n_dof) 或 (1, 6, n_dof)
            position_error: 位置误差 (3,) — 目标位置 − 当前位置（米）
            rotation_error_vec: 旋转误差 (3,) — 等效旋转轴 × 角度（弧度）

        Returns:
            Δθ (n_dof,) — 关节增量（弧度）
        """
        # 拼合 6D 误差
        error = np.concatenate([position_error, rotation_error_vec])

        # 展平可能的 batch 维度
        if jacobian.ndim == 3:
            jacobian = jacobian[0]
        if error.ndim == 2:
            error = error[0]

        # 添加 batch 维度（Isaac Sim API 习惯做法）
        jacobian_batch = jacobian[np.newaxis, :, :]
        error_batch = error[np.newaxis, :, np.newaxis]

        # ---- 四种求解方法 ------------------------------------------------

        if self.method == "singular-value-decomposition":
            U, S, Vh = np.linalg.svd(jacobian_batch)
            inv_s = np.where(self.min_singular_value < S, 1.0 / S, np.zeros_like(S))
            pseudoinverse = np.swapaxes(Vh, 1, 2)[:, :, :6] @ np.diagflat(inv_s) @ np.swapaxes(U, 1, 2)
            delta = (self.scale * pseudoinverse @ error_batch).squeeze(-1)

        elif self.method == "pseudoinverse":
            pseudoinverse = np.linalg.pinv(jacobian_batch)
            delta = (self.scale * pseudoinverse @ error_batch).squeeze(-1)

        elif self.method == "transpose":
            transpose = np.swapaxes(jacobian_batch, 1, 2)
            delta = (self.scale * transpose @ error_batch).squeeze(-1)

        elif self.method == "damped-least-squares":
            # DLS：Jᵀ(J·Jᵀ + λ²I)⁻¹ · e
            transpose = np.swapaxes(jacobian_batch, 1, 2)
            lmbda = np.eye(jacobian_batch.shape[1]) * (self.damping**2)
            delta = (
                self.scale * transpose @ np.linalg.inv(jacobian_batch @ transpose + lmbda) @ error_batch
            ).squeeze(-1)

        else:
            raise ValueError(f"不支持的 IK 方法: {self.method}")

        # 去掉 batch 维度后返回
        return delta[0]

    @staticmethod
    def compute_orientation_error(
        goal_orientation: np.ndarray, current_orientation: np.ndarray
    ) -> np.ndarray:
        """计算四元数方向误差（使用 Isaac Sim 的 Warp 四元数运算）。
        
        与 math_utils.orientation_error() 等价，但用 Warp（GPU 加速）实现。
        取四元数差值的虚部 × sign(w)，表示最小旋转的轴×角度。

        Args:
            goal_orientation: 目标方向 [w, x, y, z]
            current_orientation: 当前方向 [w, x, y, z]

        Returns:
            方向误差向量 (3,)（角度制，弧度）
        """
        goal_quat = normalize_quaternion(np.asarray(goal_orientation, dtype=np.float64))
        current_quat = normalize_quaternion(np.asarray(current_orientation, dtype=np.float64))

        # 使用 Warp 四元数运算
        goal_quat_wp = wp.from_numpy(goal_quat.astype(np.float32), dtype=wp.float32)
        current_quat_wp = wp.from_numpy(current_quat.astype(np.float32), dtype=wp.float32)

        current_quat_conjugate_wp = quaternion_conjugate(current_quat_wp)
        q_wp = quaternion_multiplication(goal_quat_wp, current_quat_conjugate_wp)
        q_np = q_wp.numpy()

        # 误差向量：取虚部，乘以符号保证短弧方向
        error = q_np[1:] * np.sign(q_np[0]) if q_np[0] != 0 else q_np[1:]
        return error.astype(np.float64)
