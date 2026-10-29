import math

import numpy as np
from isaacsim.core.experimental.prims import RigidPrim, XformPrim
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from .config import grasp_config
from .math_utils import to_numpy
from .robot import RobotArm
from .sim_app import SimulationManager


class SceneManager:
    def __init__(self, stage, target_path: str = "/World/GraspTarget",
                 target_frame_path: str = "/World/TargetFrame"):
        self._stage = stage
        self._target_path = target_path
        self._target_frame_path = target_frame_path

    @property
    def stage(self):
        return self._stage

    def create_target(
        self,
        initial_position: np.ndarray,
        initial_orientation: np.ndarray,
    ) -> None:
        cube = UsdGeom.Cube.Define(self._stage, self._target_path)
        cube.CreateSizeAttr(grasp_config.target_width)
        cube.CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.04, 0.04)])

        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*initial_position))
        xform.AddOrientOp().Set(
            Gf.Quatf(
                float(initial_orientation[0]),
                float(initial_orientation[1]),
                float(initial_orientation[2]),
                float(initial_orientation[3]),
            )
        )

        rigid_body = UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        rigid_body.CreateKinematicEnabledAttr().Set(True)
        collision = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        collision.CreateCollisionEnabledAttr().Set(False)
        UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr().Set(0.05)

        frame = UsdGeom.Xform.Define(self._stage, self._target_frame_path)
        frame_xform = UsdGeom.Xformable(frame.GetPrim())
        frame_xform.ClearXformOpOrder()
        frame_xform.AddTranslateOp().Set(Gf.Vec3d(*initial_position))
        frame_xform.AddOrientOp().Set(
            Gf.Quatf(
                float(initial_orientation[0]),
                float(initial_orientation[1]),
                float(initial_orientation[2]),
                float(initial_orientation[3]),
            )
        )

        axis_length = grasp_config.axis_length
        axis_radius = grasp_config.axis_radius
        axis_specs = (
            (
                "X",
                Gf.Vec3f(1.0, 0.0, 0.0),
                Gf.Vec3f(0.0, 90.0, 0.0),
                Gf.Vec3d(axis_length / 2, 0, 0),
            ),
            (
                "Y",
                Gf.Vec3f(0.0, 1.0, 0.0),
                Gf.Vec3f(-90.0, 0.0, 0.0),
                Gf.Vec3d(0, axis_length / 2, 0),
            ),
            (
                "Z",
                Gf.Vec3f(0.0, 0.35, 1.0),
                Gf.Vec3f(0.0, 0.0, 0.0),
                Gf.Vec3d(0, 0, axis_length / 2),
            ),
        )
        for name, color, rotation, translation in axis_specs:
            axis = UsdGeom.Cylinder.Define(
                self._stage, f"{self._target_frame_path}/{name}Axis"
            )
            axis.CreateAxisAttr("Z")
            axis.CreateRadiusAttr(axis_radius)
            axis.CreateHeightAttr(axis_length)
            axis.CreateDisplayColorAttr([color])
            axis_xform = UsdGeom.Xformable(axis.GetPrim())
            axis_xform.AddTranslateOp().Set(translation)
            axis_xform.AddRotateXYZOp().Set(rotation)

        self._target_body = RigidPrim(self._target_path)
        self._target_frame = XformPrim(self._target_frame_path)
        self._target_collision_attr = self._stage.GetPrimAtPath(
            self._target_path
        ).GetAttribute("physics:collisionEnabled")

    @property
    def target_body(self) -> RigidPrim:
        return self._target_body

    @property
    def target_frame(self) -> XformPrim:
        return self._target_frame

    @property
    def target_collision_attr(self):
        return self._target_collision_attr

    def set_target_pose(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
    ) -> None:
        sim = SimulationManager.instance()
        positions = position[np.newaxis, :].astype(np.float32)
        orientations = orientation[np.newaxis, :].astype(np.float32)
        self._target_body.set_world_poses(
            positions=positions, orientations=orientations
        )
        self._target_frame.set_world_poses(
            positions=positions, orientations=orientations
        )

        quaternion = Gf.Quatf(
            float(orientation[0]),
            float(orientation[1]),
            float(orientation[2]),
            float(orientation[3]),
        )
        for prim_path in (self._target_path, self._target_frame_path):
            prim = self._stage.GetPrimAtPath(prim_path)
            prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*position))
            prim.GetAttribute("xformOp:orient").Set(quaternion)

        for _ in range(3):
            sim.update()

        cube_position, _ = self._target_body.get_world_poses()
        frame_position, _ = self._target_frame.get_world_poses()
        sync_error = float(
            np.linalg.norm(
                to_numpy(cube_position)[0] - to_numpy(frame_position)[0]
            )
        )
        if sync_error > 1e-4:
            raise RuntimeError(
                f"方块与 XYZ marker 未同步，位置误差={sync_error:.6f} m"
            )

    def setup_camera(self) -> None:
        set_camera_view(
            eye=[2.2, 1.55, 1.55],
            target=[0.72, 0.0, 0.72],
            camera_prim_path="/OmniverseKit_Persp",
        )


def compute_non_base_link_origin_min_z(robot: RobotArm) -> float:
    ignored = {"world", "AR5_5_08L_W4C1C5_link1"}
    min_z = float("inf")
    for link_name, link_path in zip(robot.link_names, robot.link_paths[0]):
        if link_name in ignored:
            continue
        prim = RigidPrim(link_path)
        positions, _ = prim.get_world_poses()
        min_z = min(min_z, float(to_numpy(positions)[0][2]))
    return min_z


def compute_non_base_min_z(stage, robot: RobotArm) -> float:
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    min_z = float("inf")
    ignored = {"world", "AR5_5_08L_W4C1C5_link1"}
    for link_name, link_path in zip(robot.link_names, robot.link_paths[0]):
        if link_name in ignored:
            continue
        prim = stage.GetPrimAtPath(link_path)
        if not prim.IsValid():
            continue
        aligned_box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        if aligned_box.IsEmpty():
            continue
        min_z = min(min_z, float(aligned_box.GetMin()[2]))
    return min_z
