# =============================================================================
# sim_app.py — SimulationManager（Singleton）
#
# 持有 SimulationApp 引用的全局管理器。
# 跨模块统一通过 instance() 访问，避免向每个函数传递 simulation_app 参数。
# =============================================================================

# 模块级全局变量，保存唯一 SimulationManager 实例
_sim_app_instance = None


class SimulationManager:
    """封装 Isaac Sim 的 SimulationApp，提供统一的更新/查询接口。

    Singleton 模式：
      __init__() 时自动注册为全局唯一实例。
      其他模块（如 robot.py、pose_planner.py）通过 instance() 获取。

    职责：
      - setup_simulation(dt)  → 配置物理引擎步长
      - update()              → 推进一帧仿真（渲染 + 物理）
      - is_running()          → 检查窗口是否打开
      - close()               → 释放 GPU 资源
    """

    def __init__(self, simulation_app):
        """注册此实例为全局 Singleton。
        
        Args:
            simulation_app: 主入口处创建的 SimulationApp 对象
        """
        global _sim_app_instance
        self._app = simulation_app
        self._post_update_callbacks = []
        _sim_app_instance = self

    @staticmethod
    def instance():
        """获取当前唯一的 SimulationManager 实例。
        
        Raises:
            AssertionError: 如果尚未调用 __init__()
        """
        assert _sim_app_instance is not None, "SimulationManager instance not initialized."
        return _sim_app_instance

    def setup_simulation(self, dt: float) -> None:
        """配置 Isaac Sim 的物理引擎。
        
        调用底层 IsaacSim 的 SimulationManager.setup_simulation()，
        设置物理时间步长并启动物理求解器。
        
        Args:
            dt: 物理时间步长（秒），通常 1/60 ≈ 0.01667
        """
        from isaacsim.core.simulation_manager import SimulationManager as _IsaacSimMgr

        _IsaacSimMgr.setup_simulation(dt=dt)

    def update(self) -> None:
        """推进一帧仿真（渲染 + 物理 + 碰撞检测）。"""
        self._app.update()
        for callback in list(self._post_update_callbacks):
            callback()

    def add_post_update_callback(self, callback) -> None:
        """注册仿真步进后的回调。"""
        if callback not in self._post_update_callbacks:
            self._post_update_callbacks.append(callback)

    def remove_post_update_callback(self, callback) -> None:
        """移除仿真步进后的回调。"""
        if callback in self._post_update_callbacks:
            self._post_update_callbacks.remove(callback)

    def is_running(self) -> bool:
        """检查 SimulationApp 窗口是否仍在运行。"""
        return self._app.is_running()

    def close(self) -> None:
        """关闭 SimulationApp，释放 GPU 和渲染资源。"""
        self._app.close()
