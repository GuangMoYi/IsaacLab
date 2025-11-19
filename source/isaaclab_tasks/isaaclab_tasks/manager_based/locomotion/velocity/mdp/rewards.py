# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat, quat_mul, quat_conjugate, euler_zyx_from_quat
import math

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def platform_following_with_history_exp(
    env: ManagerBasedRLEnv,
    std_orientation: float = 0.15,
    std_angular_velocity: float = 0.5,
    prediction_horizon: float = 0.1,
    history_length: int = 20,
    use_god_view: bool = False,  # 是否使用"上帝视角"（直接使用当前平台状态）
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    platform_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """奖励机器狗跟随平台运动
    
    如果 use_god_view=True：直接使用当前平台姿态作为目标（上帝视角，用于对比实验）
    如果 use_god_view=False：使用神经网络预测器预测平台状态，鼓励提前响应平台运动
    
    Args:
        std_orientation: 姿态误差的标准差，用于计算奖励
        std_angular_velocity: 角速度误差的标准差（当前未使用，保留用于未来扩展）
        prediction_horizon: 预测时间范围（当前未使用，保留用于未来扩展）
        history_length: 使用的历史长度（当前未使用，保留用于未来扩展）
        use_god_view: 是否使用"上帝视角"（直接使用当前平台状态，无延迟）
    """
    robot = env.scene[robot_cfg.name]
    platform = env.scene[platform_cfg.name]
    
    # 获取机器狗当前姿态
    robot_roll, robot_pitch, _ = euler_zyx_from_quat(robot.data.root_quat_w)
    
    # 获取当前平台姿态（始终需要，作为基础）
    current_platform_roll, current_platform_pitch, _ = euler_zyx_from_quat(platform.data.root_quat_w)
    current_orientation_error = torch.sqrt(
        (current_platform_roll - robot_roll)**2 + 
        (current_platform_pitch - robot_pitch)**2 + 1e-8
    )
    
    # 根据是否使用"上帝视角"选择不同的目标
    if use_god_view:
        # 上帝视角：直接使用当前平台姿态作为目标（无延迟，用于对比实验）
        # 关键改进：即使使用"上帝视角"，也要奖励"提前响应"
        # 计算未来时刻的平台姿态（基于当前角速度预测）
        # 预测时间：0.1秒（5步，dt=0.02s）
        prediction_time = 0.1  # 预测未来0.1秒
        dt = env.step_dt if hasattr(env, 'step_dt') else 0.02
        
        # 预测未来平台姿态：current_orientation + angular_velocity * prediction_time
        future_platform_roll = current_platform_roll + platform.data.root_ang_vel_w[:, 0] * prediction_time
        future_platform_pitch = current_platform_pitch + platform.data.root_ang_vel_w[:, 1] * prediction_time
        
        # 计算未来误差：机器狗当前姿态 vs 未来平台姿态
        # 这样奖励机器狗"提前"调整姿态，而不是被动跟随
        future_orientation_error = torch.sqrt(
            (future_platform_roll - robot_roll)**2 + 
            (future_platform_pitch - robot_pitch)**2 + 1e-8
        )
        
        # 组合误差：70%未来误差（提前响应）+ 30%当前误差（基础匹配）
        # 这样机器狗会学习提前响应，而不是被动跟随
        orientation_error = 0.7 * future_orientation_error + 0.3 * current_orientation_error
    else:
        # 使用与观测空间一致的预测方法：基于t-5之前的数据预测当前时刻
        # 这样奖励函数和观测空间使用相同的预测结果，保持一致性
        platform_prediction = env.get_platform_prediction_for_observation(delay_steps=5)
        
        # 检查预测是否可用且合理（对每个环境分别处理）
        if platform_prediction is not None:
            predicted_platform_roll = platform_prediction['roll']
            predicted_platform_pitch = platform_prediction['pitch']
            
            # 对每个环境检查预测值是否合理（不是全零）
            # 如果预测值全为零，说明预测器还未训练好，使用当前值
            prediction_valid = (
                (torch.abs(predicted_platform_roll) > 1e-6) | 
                (torch.abs(predicted_platform_pitch) > 1e-6)
            )  # 形状: [num_envs]，每个环境一个布尔值
            
            # 计算预测误差（对所有环境都计算，即使预测可能无效）
            predicted_orientation_error = torch.sqrt(
                (predicted_platform_roll - robot_roll)**2 + 
                (predicted_platform_pitch - robot_pitch)**2 + 1e-8
            )
            
            # 检查预测误差是否合理（如果预测误差比当前误差大很多，说明预测不准确）
            # 如果预测误差过大，降低预测权重，主要依赖当前匹配
            prediction_ratio = predicted_orientation_error / (current_orientation_error + 1e-6)
            use_prediction = (prediction_ratio < 2.0) & prediction_valid  # 预测合理且有效
            
            # 使用torch.where根据use_prediction选择不同的权重组合
            # 预测可用且合理：预测匹配为主（70%），当前匹配为辅（30%）
            # 预测不合理或无效：主要使用当前匹配（30%预测，70%当前）
            orientation_error = torch.where(
                use_prediction,
                0.7 * predicted_orientation_error + 0.3 * current_orientation_error,  # 预测合理且有效
                0.3 * predicted_orientation_error + 0.7 * current_orientation_error   # 预测不合理或无效
            )
        else:
            # 如果预测不可用，使用当前平台姿态（基础跟随）
            orientation_error = current_orientation_error
    
    # ========== 关键改进：添加角速度匹配奖励 ==========
    # 问题：平台在运动，如果只奖励姿态匹配，机器狗总是滞后
    # 解决：奖励机器狗的角速度接近平台角速度，让机器狗"跟随"平台运动趋势
    
    # 获取机器狗和平台的角速度（roll和pitch方向）
    robot_ang_vel = robot.data.root_ang_vel_w  # [num_envs, 3]
    platform_ang_vel = platform.data.root_ang_vel_w  # [num_envs, 3]
    
    # 计算roll和pitch方向的角速度误差
    ang_vel_error_roll = torch.abs(robot_ang_vel[:, 0] - platform_ang_vel[:, 0])  # roll角速度误差
    ang_vel_error_pitch = torch.abs(robot_ang_vel[:, 1] - platform_ang_vel[:, 1])  # pitch角速度误差
    ang_vel_error = torch.sqrt(ang_vel_error_roll**2 + ang_vel_error_pitch**2 + 1e-8)  # 总角速度误差
    
    # 角速度匹配奖励：机器狗的角速度应该接近平台的角速度
    # 这样机器狗会"跟随"平台运动，而不是被动地"追赶"平台
    ang_vel_reward = torch.exp(-ang_vel_error / std_angular_velocity)
    ang_vel_reward = torch.clamp(ang_vel_reward, min=0.0, max=1.0)
    
    # ========== 姿态匹配奖励（优化版：更强调小误差） ==========
    # 问题：当前std_orientation=0.2太大，导致在误差0.065时奖励仍有0.72，梯度不够大
    # 解决：使用更小的std，并添加分段奖励，在小误差时给予更强的奖励信号
    
    # 基础奖励：使用更小的std，使奖励函数更"尖锐"
    # 当error=0时，reward=1；当error=std时，reward≈0.37
    # 减小std使奖励在小误差时下降更快，梯度更大
    orientation_reward = torch.exp(-orientation_error / std_orientation)
    
    # 分段奖励：在小误差时给予额外奖励，引导机器狗达到更高精度
    # 误差 < 0.02弧度（约1.1度）：额外奖励0.5（非常精确）
    # 误差 < 0.04弧度（约2.3度）：额外奖励0.3（精确）
    # 误差 < 0.06弧度（约3.4度）：额外奖励0.1（良好）
    very_precise_bonus = torch.where(
        orientation_error < 0.02,  # 误差小于0.02弧度（约1.1度）
        torch.ones_like(orientation_reward) * 0.5,
        torch.zeros_like(orientation_reward)
    )
    precise_bonus = torch.where(
        (orientation_error >= 0.02) & (orientation_error < 0.04),  # 误差在0.02-0.04之间
        torch.ones_like(orientation_reward) * 0.3,
        torch.zeros_like(orientation_reward)
    )
    good_bonus = torch.where(
        (orientation_error >= 0.04) & (orientation_error < 0.06),  # 误差在0.04-0.06之间
        torch.ones_like(orientation_reward) * 0.1,
        torch.zeros_like(orientation_reward)
    )
    
    # 组合奖励：基础奖励 + 分段奖励
    orientation_reward = orientation_reward + very_precise_bonus + precise_bonus + good_bonus
    orientation_reward = torch.clamp(orientation_reward, min=0.0, max=2.0)  # 最大奖励2.0（1.0基础+1.0分段奖励）
    
    # ========== 组合奖励：姿态匹配 + 角速度匹配 ==========
    # 权重分配：
    # - 姿态匹配：70%（提高权重，更强调精确的姿态匹配）
    # - 角速度匹配：30%（确保机器狗跟随平台运动趋势）
    # 这样机器狗既能保持精确的姿态匹配，又能跟随平台运动，实现"提前响应"
    combined_reward = 0.7 * orientation_reward + 0.3 * ang_vel_reward
    
    return combined_reward


# ========== 调试指标函数 ==========
def base_platform_orientation_error_metric(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    platform_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """计算机器狗基座与平台平面的姿态误差（用于评估指标）。
    
    返回roll和pitch的误差（rad）。
    值越小表示基座与平台越平行。
    """
    robot = env.scene[robot_cfg.name]
    platform = env.scene[platform_cfg.name]
    
    # 计算相对旋转
    q_rel = quat_mul(
        platform.data.root_quat_w,
        quat_conjugate(robot.data.root_quat_w)
    )
    
    # 提取roll和pitch
    rel_roll, rel_pitch, _ = euler_zyx_from_quat(q_rel)
    
    # 计算roll和pitch的误差
    orientation_error = torch.sqrt(rel_roll**2 + rel_pitch**2 + 1e-8)
    
    return orientation_error


def platform_orientation_error_metric(
    env: ManagerBasedRLEnv,
    platform_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """计算平台自身姿态误差（用于评估指标）。
    
    平台理想情况下应该是水平的（roll=0, pitch=0），
    返回平台roll和pitch的误差（rad）。
    这个值应该接近0，用于对比机器人的表现。
    """
    platform = env.scene[platform_cfg.name]
    
    # 提取平台的roll和pitch（相对于水平面）
    roll, pitch, _ = euler_zyx_from_quat(platform.data.root_quat_w)
    
    # 计算roll和pitch的误差（理想值都是0）
    orientation_error = torch.sqrt(roll**2 + pitch**2 + 1e-8)
    
    return orientation_error


def robot_relative_ang_vel_error_metric(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    platform_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
) -> torch.Tensor:
    """计算机器人相对于平台的角速度误差（用于评估指标）。
    
    返回roll和pitch角速度的误差（rad/s）。
    值越小表示角速度越同步。
    """
    robot = env.scene[robot_cfg.name]
    platform = env.scene[platform_cfg.name]
    
    # 计算相对角速度（世界坐标系）
    robot_ang_vel_w = robot.data.root_ang_vel_w
    platform_ang_vel_w = platform.data.root_ang_vel_w
    rel_ang_vel_w = robot_ang_vel_w - platform_ang_vel_w
    
    # 转换到机器人体坐标系（只考虑xy平面，即roll和pitch的角速度）
    rel_ang_vel_b = quat_apply_inverse(robot.data.root_quat_w, rel_ang_vel_w)
    
    # 计算roll和pitch角速度误差的L2范数
    ang_vel_error = torch.linalg.norm(rel_ang_vel_b[:, :2], dim=1)
    
    return ang_vel_error
