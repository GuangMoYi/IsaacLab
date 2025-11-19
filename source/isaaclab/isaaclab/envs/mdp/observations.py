# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_body_names,
    record_dtype,
    record_joint_names,
    record_joint_pos_offsets,
    record_joint_vel_offsets,
    record_shape,
)


# GMY changed: 平台观测量
from isaaclab.utils.math import euler_xyz_from_quat, euler_zyx_from_quat

# 平台角加速度（世界坐标系）
@generic_io_descriptor(
    units="rad/s^2",
    axes=["X", "Y", "Z"],
    observation_type="PlatformState",
    on_inspect=[record_shape, record_dtype],
)
def platform_ang_acc_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("platform")) -> torch.Tensor:
    """Platform angular acceleration in the simulation world frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.body_ang_acc_w[:, 0]


# 平台线加速度（世界坐标系）
@generic_io_descriptor(
    units="m/s^2",
    axes=["X", "Y", "Z"],
    observation_type="PlatformState",
    on_inspect=[record_shape, record_dtype],
)
def platform_lin_acc_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("platform")) -> torch.Tensor:
    """Platform linear acceleration in the simulation world frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.body_lin_acc_w[:, 0]


# 平台线速度（世界坐标系）
@generic_io_descriptor(
    units="m/s",
    axes=["X", "Y", "Z"],
    observation_type="PlatformState",
    on_inspect=[record_shape, record_dtype],
)
def platform_lin_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("platform")) -> torch.Tensor:
    """Platform linear velocity in the simulation world frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w


# 平台角速度（世界坐标系）
@generic_io_descriptor(
    units="rad/s",
    axes=["X", "Y", "Z"],
    observation_type="PlatformState",
    on_inspect=[record_shape, record_dtype],
)
def platform_ang_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("platform")) -> torch.Tensor:
    """Platform angular velocity in the simulation world frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w


# 平台位置（世界坐标系）
@generic_io_descriptor(
    units="m",
    axes=["X", "Y", "Z"],
    observation_type="PlatformState",
    on_inspect=[record_shape, record_dtype],
)
def platform_pos_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("platform")) -> torch.Tensor:
    """Platform position in the simulation world frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w


# 平台姿态（欧拉角，世界坐标系）
@generic_io_descriptor(
    units="rad",
    axes=["Roll", "Pitch", "Yaw"],
    observation_type="PlatformState",
    on_inspect=[record_shape, record_dtype],
)
def platform_ang_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("platform")) -> torch.Tensor:
    """Platform orientation (Euler angles) in the simulation world frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.stack([roll, pitch, yaw], dim=-1).to(
        device=asset.data.root_quat_w.device,
        dtype=asset.data.root_quat_w.dtype,
    )

## GMY changed: 相对静止==观测值
# 平台线速度（在机器人体坐标系下）
@generic_io_descriptor(
    units="m/s",
    axes=["X", "Y", "Z"],
    observation_type="PlatformState",
    on_inspect=[record_shape, record_dtype],
)
def platform_lin_vel_b(
    env: ManagerBasedEnv,
    platform_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Platform linear velocity in robot body frame."""
    platform: RigidObject = env.scene[platform_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    # 平台世界坐标系速度
    platform_vel_w = platform.data.root_lin_vel_w
    # 转换到机器人体坐标系
    platform_vel_b = math_utils.quat_apply_inverse(robot.data.root_quat_w, platform_vel_w)
    return platform_vel_b


# 平台角速度（在机器人体坐标系下）
@generic_io_descriptor(
    units="rad/s",
    axes=["X", "Y", "Z"],
    observation_type="PlatformState",
    on_inspect=[record_shape, record_dtype],
)
def platform_ang_vel_b(
    env: ManagerBasedEnv,
    platform_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Platform angular velocity in robot body frame."""
    platform: RigidObject = env.scene[platform_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    # 平台世界坐标系角速度
    platform_ang_vel_w = platform.data.root_ang_vel_w
    # 转换到机器人体坐标系
    platform_ang_vel_b = math_utils.quat_apply_inverse(robot.data.root_quat_w, platform_ang_vel_w)
    return platform_ang_vel_b


# 平台姿态（欧拉角，在机器人体坐标系下）
@generic_io_descriptor(
    units="rad",
    axes=["Roll", "Pitch", "Yaw"],
    observation_type="PlatformState",
    on_inspect=[record_shape, record_dtype],
)
def platform_ang_b(
    env: ManagerBasedEnv,
    platform_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Platform orientation (Euler angles) in robot body frame."""
    platform: RigidObject = env.scene[platform_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    # 计算相对旋转：platform相对于robot的旋转
    # q_rel = q_platform * q_robot^-1
    q_rel = math_utils.quat_mul(
        platform.data.root_quat_w,
        math_utils.quat_conjugate(robot.data.root_quat_w)
    )
    roll, pitch, yaw = euler_xyz_from_quat(q_rel)
    return torch.stack([roll, pitch, yaw], dim=-1).to(
        device=platform.data.root_quat_w.device,
        dtype=platform.data.root_quat_w.dtype,
    )


# 机器人相对于平台的线速度（在机器人体坐标系下）
@generic_io_descriptor(
    units="m/s",
    axes=["X", "Y", "Z"],
    observation_type="RelativeVelocity",
    on_inspect=[record_shape, record_dtype],
)
def robot_relative_lin_vel_to_platform(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    platform_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """Robot relative linear velocity to platform in robot body frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    platform: RigidObject = env.scene[platform_cfg.name]
    
    # 机器狗和平台的世界坐标系速度
    robot_vel_w = robot.data.root_lin_vel_w
    platform_vel_w = platform.data.root_lin_vel_w
    
    # 计算相对速度（世界坐标系）
    rel_vel_w = robot_vel_w - platform_vel_w
    
    # 转换到机器人体坐标系
    rel_vel_b = math_utils.quat_apply_inverse(robot.data.root_quat_w, rel_vel_w)
    
    return rel_vel_b


# 机器人相对于平台的角速度（在机器人体坐标系下）
@generic_io_descriptor(
    units="rad/s",
    axes=["X", "Y", "Z"],
    observation_type="RelativeVelocity",
    on_inspect=[record_shape, record_dtype],
)
def orientation_error_ratio_metric(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    platform_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """计算机器狗误差和平台误差的比值（用于评估跟随效果）。
    
    比值 = 机器狗基座与平台姿态误差 / 平台自身姿态误差
    比值越小，说明机器狗跟随效果越好（机器狗误差相对于平台误差很小）
    """
    robot = env.scene[robot_cfg.name]
    platform = env.scene[platform_cfg.name]
    
    # 计算机器狗基座与平台姿态误差
    q_rel = math_utils.quat_mul(
        platform.data.root_quat_w,
        math_utils.quat_conjugate(robot.data.root_quat_w)
    )
    rel_roll, rel_pitch, _ = euler_zyx_from_quat(q_rel)
    robot_platform_error = torch.sqrt(rel_roll**2 + rel_pitch**2 + 1e-8)
    
    # 计算平台自身姿态误差
    platform_roll, platform_pitch, _ = euler_zyx_from_quat(platform.data.root_quat_w)
    platform_error = torch.sqrt(platform_roll**2 + platform_pitch**2 + 1e-8)
    
    # 计算比值（避免除零）
    ratio = robot_platform_error / (platform_error + 1e-8)
    
    return ratio


def robot_relative_ang_vel_to_platform(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    platform_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """Robot relative angular velocity to platform in robot body frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    platform: RigidObject = env.scene[platform_cfg.name]
    
    # 机器狗和平台的世界坐标系角速度
    robot_ang_vel_w = robot.data.root_ang_vel_w
    platform_ang_vel_w = platform.data.root_ang_vel_w
    
    # 计算相对角速度（世界坐标系）
    rel_ang_vel_w = robot_ang_vel_w - platform_ang_vel_w
    
    # 转换到机器人体坐标系
    rel_ang_vel_b = math_utils.quat_apply_inverse(robot.data.root_quat_w, rel_ang_vel_w)
    
    return rel_ang_vel_b


# 机器人相对于平台的位置（在平台坐标系下，只考虑xy平面）
@generic_io_descriptor(
    units="m",
    axes=["X", "Y"],
    observation_type="RelativePosition",
    on_inspect=[record_shape, record_dtype],
)
def robot_relative_pos_to_platform_xy(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    platform_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """Robot relative position to platform in platform frame (xy plane only)."""
    robot: Articulation = env.scene[robot_cfg.name]
    platform: RigidObject = env.scene[platform_cfg.name]
    
    # 机器狗和平台的世界坐标系位置
    robot_pos_w = robot.data.root_pos_w
    platform_pos_w = platform.data.root_pos_w
    
    # 计算相对位置（世界坐标系）
    rel_pos_w = robot_pos_w - platform_pos_w
    
    # 转换到平台坐标系（只考虑xy平面）
    platform_quat_inv = math_utils.quat_conjugate(platform.data.root_quat_w)
    rel_pos_p = math_utils.quat_apply(platform_quat_inv, rel_pos_w)
    
    # 只返回xy平面
    return rel_pos_p[:, :2]


# 机器人线速度（世界坐标系，用于监控机器狗运动）
@generic_io_descriptor(
    units="m/s",
    axes=["X", "Y", "Z"],
    observation_type="RobotState",
    on_inspect=[record_shape, record_dtype],
)
def robot_lin_vel_w(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robot linear velocity in world frame (for monitoring robot movement)."""
    robot: Articulation = env.scene[robot_cfg.name]
    return robot.data.root_lin_vel_w


# 机器人角速度（世界坐标系，用于监控机器狗运动）
@generic_io_descriptor(
    units="rad/s",
    axes=["X", "Y", "Z"],
    observation_type="RobotState",
    on_inspect=[record_shape, record_dtype],
)
def robot_ang_vel_w(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robot angular velocity in world frame (for monitoring robot movement)."""
    robot: Articulation = env.scene[robot_cfg.name]
    return robot.data.root_ang_vel_w


# 机器人线速度大小（用于监控机器狗是否在运动）
@generic_io_descriptor(
    units="m/s",
    axes=["Magnitude"],
    observation_type="RobotState",
    on_inspect=[record_shape, record_dtype],
)
def robot_lin_vel_magnitude(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robot linear velocity magnitude (for monitoring if robot is moving)."""
    robot: Articulation = env.scene[robot_cfg.name]
    lin_vel = robot.data.root_lin_vel_w
    return torch.linalg.norm(lin_vel, dim=1, keepdim=True)


# ========== 平台历史数据和预测观测（用于机器狗学习跟随平台） ==========
# 平台历史姿态（t-5之前的roll和pitch，展平为向量）
@generic_io_descriptor(
    units="rad",
    observation_type="PlatformHistory",
    on_inspect=[record_shape, record_dtype],
)
def platform_history_orientation(
    env: ManagerBasedEnv,
    delay_steps: int = 5,
    history_length: int = 10,
    platform_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """平台历史姿态（t-delay_steps之前的roll和pitch）
    
    返回展平的历史数据：[roll_history, pitch_history]，形状为 [num_envs, history_length*2]
    """
    if not hasattr(env, 'get_platform_delayed_history'):
        # 如果环境不支持，返回零
        num_envs = env.scene[platform_cfg.name].data.root_quat_w.shape[0]
        device = env.scene[platform_cfg.name].data.root_quat_w.device
        return torch.zeros(num_envs, history_length * 2, device=device)
    
    history_data = env.get_platform_delayed_history(delay_steps=delay_steps, history_length=history_length)
    
    if history_data is None:
        num_envs = env.scene[platform_cfg.name].data.root_quat_w.shape[0]
        device = env.scene[platform_cfg.name].data.root_quat_w.device
        return torch.zeros(num_envs, history_length * 2, device=device)
    
    # 展平历史数据：[roll_history, pitch_history]
    history_roll = history_data['roll']  # [num_envs, history_length]
    history_pitch = history_data['pitch']  # [num_envs, history_length]
    
    # 拼接为 [num_envs, history_length*2]
    return torch.cat([history_roll, history_pitch], dim=1)


# 平台历史角速度（t-5之前的roll和pitch角速度，展平为向量）
@generic_io_descriptor(
    units="rad/s",
    observation_type="PlatformHistory",
    on_inspect=[record_shape, record_dtype],
)
def platform_history_angular_velocity(
    env: ManagerBasedEnv,
    delay_steps: int = 5,
    history_length: int = 10,
    platform_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """平台历史角速度（t-delay_steps之前的roll和pitch角速度）
    
    返回展平的历史数据：[roll_ang_vel_history, pitch_ang_vel_history]，形状为 [num_envs, history_length*2]
    """
    if not hasattr(env, 'get_platform_delayed_history'):
        num_envs = env.scene[platform_cfg.name].data.root_quat_w.shape[0]
        device = env.scene[platform_cfg.name].data.root_quat_w.device
        return torch.zeros(num_envs, history_length * 2, device=device)
    
    history_data = env.get_platform_delayed_history(delay_steps=delay_steps, history_length=history_length)
    
    if history_data is None:
        num_envs = env.scene[platform_cfg.name].data.root_quat_w.shape[0]
        device = env.scene[platform_cfg.name].data.root_quat_w.device
        return torch.zeros(num_envs, history_length * 2, device=device)
    
    # 展平历史数据：[roll_ang_vel_history, pitch_ang_vel_history]
    history_roll_ang_vel = history_data['roll_ang_vel']  # [num_envs, history_length]
    history_pitch_ang_vel = history_data['pitch_ang_vel']  # [num_envs, history_length]
    
    # 拼接为 [num_envs, history_length*2]
    return torch.cat([history_roll_ang_vel, history_pitch_ang_vel], dim=1)


# 预测的当前平台姿态（基于t-5之前的数据预测当前时刻）
@generic_io_descriptor(
    units="rad",
    axes=["Roll", "Pitch"],
    observation_type="PlatformPrediction",
    on_inspect=[record_shape, record_dtype],
)
def platform_predicted_orientation(
    env: ManagerBasedEnv,
    delay_steps: int = 5,
    platform_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """预测的当前平台姿态（基于t-delay_steps之前的数据预测当前时刻）
    
    返回预测的roll和pitch：[predicted_roll, predicted_pitch]，形状为 [num_envs, 2]
    """
    if not hasattr(env, 'get_platform_prediction_for_observation'):
        num_envs = env.scene[platform_cfg.name].data.root_quat_w.shape[0]
        device = env.scene[platform_cfg.name].data.root_quat_w.device
        return torch.zeros(num_envs, 2, device=device)
    
    prediction = env.get_platform_prediction_for_observation(delay_steps=delay_steps)
    
    if prediction is None:
        num_envs = env.scene[platform_cfg.name].data.root_quat_w.shape[0]
        device = env.scene[platform_cfg.name].data.root_quat_w.device
        return torch.zeros(num_envs, 2, device=device)
    
    # 返回预测的roll和pitch
    predicted_roll = prediction['roll']  # [num_envs]
    predicted_pitch = prediction['pitch']  # [num_envs]
    
    return torch.stack([predicted_roll, predicted_pitch], dim=1)  # [num_envs, 2]


# 预测的当前平台角速度（基于t-5之前的数据预测当前时刻）
@generic_io_descriptor(
    units="rad/s",
    axes=["Roll", "Pitch"],
    observation_type="PlatformPrediction",
    on_inspect=[record_shape, record_dtype],
)
def platform_predicted_angular_velocity(
    env: ManagerBasedEnv,
    delay_steps: int = 5,
    platform_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """预测的当前平台角速度（基于t-delay_steps之前的数据预测当前时刻）
    
    返回预测的roll和pitch角速度：[predicted_roll_ang_vel, predicted_pitch_ang_vel]，形状为 [num_envs, 2]
    """
    if not hasattr(env, 'get_platform_prediction_for_observation'):
        num_envs = env.scene[platform_cfg.name].data.root_quat_w.shape[0]
        device = env.scene[platform_cfg.name].data.root_quat_w.device
        return torch.zeros(num_envs, 2, device=device)
    
    prediction = env.get_platform_prediction_for_observation(delay_steps=delay_steps)
    
    if prediction is None:
        num_envs = env.scene[platform_cfg.name].data.root_quat_w.shape[0]
        device = env.scene[platform_cfg.name].data.root_quat_w.device
        return torch.zeros(num_envs, 2, device=device)
    
    # 返回预测的roll和pitch角速度
    predicted_roll_ang_vel = prediction['roll_ang_vel']  # [num_envs]
    predicted_pitch_ang_vel = prediction['pitch_ang_vel']  # [num_envs]
    
    return torch.stack([predicted_roll_ang_vel, predicted_pitch_ang_vel], dim=1)  # [num_envs, 2]


# ============================================================================
# 上帝视角：直接观测当前平台状态（无延迟，用于对比实验）
# ============================================================================

# 当前平台姿态（上帝视角，直接使用当前时刻的平台姿态）
@generic_io_descriptor(
    units="rad",
    axes=["Roll", "Pitch"],
    observation_type="PlatformState",
    on_inspect=[record_shape, record_dtype],
)
def platform_current_orientation(
    env: ManagerBasedEnv,
    platform_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """当前平台姿态（上帝视角，直接使用当前时刻的平台roll和pitch）
    
    返回当前平台的roll和pitch：[current_roll, current_pitch]，形状为 [num_envs, 2]
    用于对比实验：如果机器狗能直接观测到当前平台状态，能否学会跟随
    """
    platform: RigidObject = env.scene[platform_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(platform.data.root_quat_w)
    return torch.stack([roll, pitch], dim=1)  # [num_envs, 2]


# 当前平台角速度（上帝视角，直接使用当前时刻的平台角速度）
@generic_io_descriptor(
    units="rad/s",
    axes=["Roll", "Pitch"],
    observation_type="PlatformState",
    on_inspect=[record_shape, record_dtype],
)
def platform_current_angular_velocity(
    env: ManagerBasedEnv,
    platform_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """当前平台角速度（上帝视角，直接使用当前时刻的平台roll和pitch角速度）
    
    返回当前平台的roll和pitch角速度：[current_roll_ang_vel, current_pitch_ang_vel]，形状为 [num_envs, 2]
    用于对比实验：如果机器狗能直接观测到当前平台状态，能否学会跟随
    """
    platform: RigidObject = env.scene[platform_cfg.name]
    ang_vel = platform.data.root_ang_vel_w
    return torch.stack([ang_vel[:, 0], ang_vel[:, 1]], dim=1)  # [num_envs, 2] (roll, pitch)
# ============================================================================

# 机器人角速度大小（用于监控机器狗是否在运动）
@generic_io_descriptor(
    units="rad/s",
    axes=["Magnitude"],
    observation_type="RobotState",
    on_inspect=[record_shape, record_dtype],
)
def robot_ang_vel_magnitude(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robot angular velocity magnitude (for monitoring if robot is moving)."""
    robot: Articulation = env.scene[robot_cfg.name]
    ang_vel = robot.data.root_ang_vel_w
    return torch.linalg.norm(ang_vel, dim=1, keepdim=True)

"""
Root state.
"""


@generic_io_descriptor(units="m", axes=["Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype])
def base_pos_z(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root height in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2].unsqueeze(-1)


@generic_io_descriptor(
    units="m/s", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def base_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


@generic_io_descriptor(
    units="rad/s", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def base_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


@generic_io_descriptor(
    units="m/s^2", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gravity projection on the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


@generic_io_descriptor(
    units="m", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def root_pos_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root position in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


@generic_io_descriptor(
    units="unit", axes=["W", "X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def root_quat_w(
    env: ManagerBasedEnv, make_quat_unique: bool = False, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset root orientation (w, x, y, z) in the environment frame.

    If :attr:`make_quat_unique` is True, then returned quaternion is made unique by ensuring
    the quaternion has non-negative real component. This is because both ``q`` and ``-q`` represent
    the same orientation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    quat = asset.data.root_quat_w
    # make the quaternion real-part positive if configured
    return math_utils.quat_unique(quat) if make_quat_unique else quat


@generic_io_descriptor(
    units="m/s", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def root_lin_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root linear velocity in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w


@generic_io_descriptor(
    units="rad/s", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def root_ang_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root angular velocity in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w


"""
Body state
"""


@generic_io_descriptor(observation_type="BodyState", on_inspect=[record_shape, record_dtype, record_body_names])
def body_pose_w(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The flattened body poses of the asset w.r.t the env.scene.origin.

    Note: Only the bodies configured in :attr:`asset_cfg.body_ids` will have their poses returned.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with this observation.

    Returns:
        The poses of bodies in articulation [num_env, 7 * num_bodies]. Pose order is [x,y,z,qw,qx,qy,qz].
        Output is stacked horizontally per body.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # access the body poses in world frame
    pose = asset.data.body_pose_w[:, asset_cfg.body_ids, :7]
    pose[..., :3] = pose[..., :3] - env.scene.env_origins.unsqueeze(1)
    return pose.reshape(env.num_envs, -1)


@generic_io_descriptor(observation_type="BodyState", on_inspect=[record_shape, record_dtype, record_body_names])
def body_projected_gravity_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The direction of gravity projected on to bodies of an Articulation.

    Note: Only the bodies configured in :attr:`asset_cfg.body_ids` will have their poses returned.

    Args:
        env: The environment.
        asset_cfg: The Articulation associated with this observation.

    Returns:
        The unit vector direction of gravity projected onto body_name's frame. Gravity projection vector order is
        [x,y,z]. Output is stacked horizontally per body.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]
    gravity_dir = asset.data.GRAVITY_VEC_W.unsqueeze(1)
    return math_utils.quat_apply_inverse(body_quat, gravity_dir).view(env.num_envs, -1)


"""
Joint state.
"""


@generic_io_descriptor(
    observation_type="JointState", on_inspect=[record_joint_names, record_dtype, record_shape], units="rad"
)
def joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]


@generic_io_descriptor(
    observation_type="JointState",
    on_inspect=[record_joint_names, record_dtype, record_shape, record_joint_pos_offsets],
    units="rad",
)
def joint_pos_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]


@generic_io_descriptor(observation_type="JointState", on_inspect=[record_joint_names, record_dtype, record_shape])
def joint_pos_limit_normalized(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The joint positions of the asset normalized with the asset's joint limits.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their normalized positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return math_utils.scale_transform(
        asset.data.joint_pos[:, asset_cfg.joint_ids],
        asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0],
        asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1],
    )


@generic_io_descriptor(
    observation_type="JointState", on_inspect=[record_joint_names, record_dtype, record_shape], units="rad/s"
)
def joint_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]


@generic_io_descriptor(
    observation_type="JointState",
    on_inspect=[record_joint_names, record_dtype, record_shape, record_joint_vel_offsets],
    units="rad/s",
)
def joint_vel_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]


@generic_io_descriptor(
    observation_type="JointState", on_inspect=[record_joint_names, record_dtype, record_shape], units="N.m"
)
def joint_effort(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint applied effort of the robot.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their effort returned.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with this observation.

    Returns:
        The joint effort (N or N-m) for joint_names in asset_cfg, shape is [num_env,num_joints].
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids]


"""
Sensors.
"""


def height_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset


def body_incoming_wrench(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Incoming spatial wrench on bodies of an articulation in the simulation world frame.

    This is the 6-D wrench (force and torque) applied to the body link by the incoming joint force.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # obtain the link incoming forces in world frame
    body_incoming_joint_wrench_b = asset.data.body_incoming_joint_wrench_b[:, asset_cfg.body_ids]
    return body_incoming_joint_wrench_b.view(env.num_envs, -1)


def imu_orientation(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Imu sensor orientation in the simulation world frame.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an IMU sensor. Defaults to SceneEntityCfg("imu").

    Returns:
        Orientation in the world frame in (w, x, y, z) quaternion form. Shape is (num_envs, 4).
    """
    # extract the used quantities (to enable type-hinting)
    asset: Imu = env.scene[asset_cfg.name]
    # return the orientation quaternion
    return asset.data.quat_w


def imu_projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Imu sensor orientation w.r.t the env.scene.origin.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an Imu sensor.

    Returns:
        Gravity projected on imu_frame, shape of torch.tensor is (num_env,3).
    """

    asset: Imu = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


def imu_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Imu sensor angular velocity w.r.t. environment origin expressed in the sensor frame.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an IMU sensor. Defaults to SceneEntityCfg("imu").

    Returns:
        The angular velocity (rad/s) in the sensor frame. Shape is (num_envs, 3).
    """
    # extract the used quantities (to enable type-hinting)
    asset: Imu = env.scene[asset_cfg.name]
    # return the angular velocity
    return asset.data.ang_vel_b


def imu_lin_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Imu sensor linear acceleration w.r.t. the environment origin expressed in sensor frame.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an IMU sensor. Defaults to SceneEntityCfg("imu").

    Returns:
        The linear acceleration (m/s^2) in the sensor frame. Shape is (num_envs, 3).
    """
    asset: Imu = env.scene[asset_cfg.name]
    return asset.data.lin_acc_b


def image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor.

    If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
    data-types:

    - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
    - "depth" or "distance_to_camera" or "distance_to_plane": Replaces infinity values with zero.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The data type to pull from the desired camera. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        normalize: Whether to normalize the images. This depends on the selected data type.
            Defaults to True.

    Returns:
        The images produced at the last time-step
    """
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # obtain the input image
    images = sensor.data.output[data_type]

    # depth image conversion
    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

    # rgb/depth/normals image normalization
    if normalize:
        if data_type == "rgb":
            images = images.float() / 255.0
            mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
            images -= mean_tensor
        elif "distance_to" in data_type or "depth" in data_type:
            images[images == float("inf")] = 0
        elif "normals" in data_type:
            images = (images + 1.0) * 0.5

    return images.clone()


class image_features(ManagerTermBase):
    """Extracted image features from a pre-trained frozen encoder.

    This term uses models from the model zoo in PyTorch and extracts features from the images.

    It calls the :func:`image` function to get the images and then processes them using the model zoo.

    A user can provide their own model zoo configuration to use different models for feature extraction.
    The model zoo configuration should be a dictionary that maps different model names to a dictionary
    that defines the model, preprocess and inference functions. The dictionary should have the following
    entries:

    - "model": A callable that returns the model when invoked without arguments.
    - "reset": A callable that resets the model. This is useful when the model has a state that needs to be reset.
    - "inference": A callable that, when given the model and the images, returns the extracted features.

    If the model zoo configuration is not provided, the default model zoo configurations are used. The default
    model zoo configurations include the models from Theia :cite:`shang2024theia` and ResNet :cite:`he2016deep`.
    These models are loaded from `Hugging-Face transformers <https://huggingface.co/docs/transformers/index>`_ and
    `PyTorch torchvision <https://pytorch.org/vision/stable/models.html>`_ respectively.

    Args:
        sensor_cfg: The sensor configuration to poll. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The sensor data type. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        model_zoo_cfg: A user-defined dictionary that maps different model names to their respective configurations.
            Defaults to None. If None, the default model zoo configurations are used.
        model_name: The name of the model to use for inference. Defaults to "resnet18".
        model_device: The device to store and infer the model on. This is useful when offloading the computation
            from the environment simulation device. Defaults to the environment device.
        inference_kwargs: Additional keyword arguments to pass to the inference function. Defaults to None,
            which means no additional arguments are passed.

    Returns:
        The extracted features tensor. Shape is (num_envs, feature_dim).

    Raises:
        ValueError: When the model name is not found in the provided model zoo configuration.
        ValueError: When the model name is not found in the default model zoo configuration.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        # extract parameters from the configuration
        self.model_zoo_cfg: dict = cfg.params.get("model_zoo_cfg")  # type: ignore
        self.model_name: str = cfg.params.get("model_name", "resnet18")  # type: ignore
        self.model_device: str = cfg.params.get("model_device", env.device)  # type: ignore

        # List of Theia models - These are configured through `_prepare_theia_transformer_model` function
        default_theia_models = [
            "theia-tiny-patch16-224-cddsv",
            "theia-tiny-patch16-224-cdiv",
            "theia-small-patch16-224-cdiv",
            "theia-base-patch16-224-cdiv",
            "theia-small-patch16-224-cddsv",
            "theia-base-patch16-224-cddsv",
        ]
        # List of ResNet models - These are configured through `_prepare_resnet_model` function
        default_resnet_models = ["resnet18", "resnet34", "resnet50", "resnet101"]

        # Check if model name is specified in the model zoo configuration
        if self.model_zoo_cfg is not None and self.model_name not in self.model_zoo_cfg:
            raise ValueError(
                f"Model name '{self.model_name}' not found in the provided model zoo configuration."
                " Please add the model to the model zoo configuration or use a different model name."
                f" Available models in the provided list: {list(self.model_zoo_cfg.keys())}."
                "\nHint: If you want to use a default model, consider using one of the following models:"
                f" {default_theia_models + default_resnet_models}. In this case, you can remove the"
                " 'model_zoo_cfg' parameter from the observation term configuration."
            )
        if self.model_zoo_cfg is None:
            if self.model_name in default_theia_models:
                model_config = self._prepare_theia_transformer_model(self.model_name, self.model_device)
            elif self.model_name in default_resnet_models:
                model_config = self._prepare_resnet_model(self.model_name, self.model_device)
            else:
                raise ValueError(
                    f"Model name '{self.model_name}' not found in the default model zoo configuration."
                    f" Available models: {default_theia_models + default_resnet_models}."
                )
        else:
            model_config = self.model_zoo_cfg[self.model_name]

        # Retrieve the model, preprocess and inference functions
        self._model = model_config["model"]()
        self._reset_fn = model_config.get("reset")
        self._inference_fn = model_config["inference"]

    def reset(self, env_ids: torch.Tensor | None = None):
        # reset the model if a reset function is provided
        # this might be useful when the model has a state that needs to be reset
        # for example: video transformers
        if self._reset_fn is not None:
            self._reset_fn(self._model, env_ids)

    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
        data_type: str = "rgb",
        convert_perspective_to_orthogonal: bool = False,
        model_zoo_cfg: dict | None = None,
        model_name: str = "resnet18",
        model_device: str | None = None,
        inference_kwargs: dict | None = None,
    ) -> torch.Tensor:
        # obtain the images from the sensor
        image_data = image(
            env=env,
            sensor_cfg=sensor_cfg,
            data_type=data_type,
            convert_perspective_to_orthogonal=convert_perspective_to_orthogonal,
            normalize=False,  # we pre-process based on model
        )
        # store the device of the image
        image_device = image_data.device
        # forward the images through the model
        features = self._inference_fn(self._model, image_data, **(inference_kwargs or {}))

        # move the features back to the image device
        return features.detach().to(image_device)

    """
    Helper functions.
    """

    def _prepare_theia_transformer_model(self, model_name: str, model_device: str) -> dict:
        """Prepare the Theia transformer model for inference.

        Args:
            model_name: The name of the Theia transformer model to prepare.
            model_device: The device to store and infer the model on.

        Returns:
            A dictionary containing the model and inference functions.
        """
        from transformers import AutoModel

        def _load_model() -> torch.nn.Module:
            """Load the Theia transformer model."""
            model = AutoModel.from_pretrained(f"theaiinstitute/{model_name}", trust_remote_code=True).eval()
            return model.to(model_device)

        def _inference(model, images: torch.Tensor) -> torch.Tensor:
            """Inference the Theia transformer model.

            Args:
                model: The Theia transformer model.
                images: The preprocessed image tensor. Shape is (num_envs, height, width, channel).

            Returns:
                The extracted features tensor. Shape is (num_envs, feature_dim).
            """
            # Move the image to the model device
            image_proc = images.to(model_device)
            # permute the image to (num_envs, channel, height, width)
            image_proc = image_proc.permute(0, 3, 1, 2).float() / 255.0
            # Normalize the image
            mean = torch.tensor([0.485, 0.456, 0.406], device=model_device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=model_device).view(1, 3, 1, 1)
            image_proc = (image_proc - mean) / std

            # Taken from Transformers; inference converted to be GPU only
            features = model.backbone.model(pixel_values=image_proc, interpolate_pos_encoding=True)
            return features.last_hidden_state[:, 1:]

        # return the model, preprocess and inference functions
        return {"model": _load_model, "inference": _inference}

    def _prepare_resnet_model(self, model_name: str, model_device: str) -> dict:
        """Prepare the ResNet model for inference.

        Args:
            model_name: The name of the ResNet model to prepare.
            model_device: The device to store and infer the model on.

        Returns:
            A dictionary containing the model and inference functions.
        """
        from torchvision import models

        def _load_model() -> torch.nn.Module:
            """Load the ResNet model."""
            # map the model name to the weights
            resnet_weights = {
                "resnet18": "ResNet18_Weights.IMAGENET1K_V1",
                "resnet34": "ResNet34_Weights.IMAGENET1K_V1",
                "resnet50": "ResNet50_Weights.IMAGENET1K_V1",
                "resnet101": "ResNet101_Weights.IMAGENET1K_V1",
            }

            # load the model
            model = getattr(models, model_name)(weights=resnet_weights[model_name]).eval()
            return model.to(model_device)

        def _inference(model, images: torch.Tensor) -> torch.Tensor:
            """Inference the ResNet model.

            Args:
                model: The ResNet model.
                images: The preprocessed image tensor. Shape is (num_envs, channel, height, width).

            Returns:
                The extracted features tensor. Shape is (num_envs, feature_dim).
            """
            # move the image to the model device
            image_proc = images.to(model_device)
            # permute the image to (num_envs, channel, height, width)
            image_proc = image_proc.permute(0, 3, 1, 2).float() / 255.0
            # normalize the image
            mean = torch.tensor([0.485, 0.456, 0.406], device=model_device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=model_device).view(1, 3, 1, 1)
            image_proc = (image_proc - mean) / std

            # forward the image through the model
            return model(image_proc)

        # return the model, preprocess and inference functions
        return {"model": _load_model, "inference": _inference}


"""
Actions.
"""


@generic_io_descriptor(dtype=torch.float32, observation_type="Action", on_inspect=[record_shape])
def last_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    if action_name is None:
        return env.action_manager.action
    else:
        return env.action_manager.get_term(action_name).raw_actions


"""
Commands.
"""


@generic_io_descriptor(dtype=torch.float32, observation_type="Command", on_inspect=[record_shape])
def generated_commands(env: ManagerBasedRLEnv, command_name: str | None = None) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    return env.command_manager.get_command(command_name)


"""
Time.
"""


def current_time_s(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The current time in the episode (in seconds)."""
    return env.episode_length_buf.unsqueeze(1) * env.step_dt


def remaining_time_s(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The maximum time remaining in the episode (in seconds)."""
    return env.max_episode_length_s - env.episode_length_buf.unsqueeze(1) * env.step_dt
