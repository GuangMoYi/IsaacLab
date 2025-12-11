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
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import string as string_utils
from isaaclab.utils.math import quat_apply_inverse, yaw_quat, quat_mul, quat_conjugate, euler_zyx_from_quat
from isaaclab.assets import Articulation
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
    
    # 初始化目标姿态变量（用于方向正确性计算）
    target_roll = current_platform_roll
    target_pitch = current_platform_pitch
    
    # 根据是否使用"上帝视角"和是否使用推断模式选择不同的目标
    use_inference = getattr(env, 'use_platform_inference', False)
    
    if use_god_view:
        # 上帝视角：直接使用当前平台姿态作为目标（无延迟，用于对比实验）
        # 关键改进：即使使用"上帝视角"，也要奖励"提前响应"
        # 使用神经网络预测未来时刻的平台姿态（基于历史运动模式，而不是简单的线性外推）
        # 预测时间：0.1秒（5步，dt=0.02s）
        prediction_time = 0.1  # 预测未来0.1秒
        
        # ========== 关键改进：如果使用推断模式，优先使用傅里叶/多项式外推 ==========
        if use_inference:
            # 使用推断模式：从观测信息推断平台运动，然后使用傅里叶/多项式外推预测未来
            # 先检查是否有推断的历史数据
            if hasattr(env, '_inferred_platform_history') and len(env._inferred_platform_history.get('roll', [])) >= 2:
                # 使用推断的历史数据进行傅里叶/多项式外推
                future_platform_roll, future_platform_pitch = env._extrapolate_from_inferred_history(
                    prediction_time=prediction_time,
                    history_window=50
                )
                future_orientation_error = torch.sqrt(
                    (future_platform_roll - robot_roll)**2 + 
                    (future_platform_pitch - robot_pitch)**2 + 1e-8
                )
            else:
                # 如果推断历史数据不足，回退到直接观测平台并使用外推
                future_platform_roll, future_platform_pitch = env.extrapolate_platform_future_advanced(
                    prediction_time=prediction_time,
                    history_window=15
                )
                future_orientation_error = torch.sqrt(
                    (future_platform_roll - robot_roll)**2 + 
                    (future_platform_pitch - robot_pitch)**2 + 1e-8
                )
        else:
            # 不使用推断模式：使用直接观测的平台数据
            # ========== 关键改进：根据预测质量决定使用哪种预测方法 ==========
            # 只有当预测质量足够好时，才使用神经网络预测；否则使用简单的线性外推
            # 这样可以避免在预测器还未训练好时使用不准确的预测
            
            # 检查预测质量是否足够好
            prediction_quality_good = env.is_platform_prediction_quality_good()
            
            # 获取预测质量详细信息
            quality_info = env.get_platform_prediction_quality_info()
            
            # 打印是否使用神经网络预测（每隔一定步数打印一次，避免打印太频繁）
            if not hasattr(env, '_last_prediction_method_print_step'):
                env._last_prediction_method_print_step = -1
                env._last_prediction_method = None
            
            # 每隔500步打印一次，或者当预测方法改变时打印（降低打印间隔，确保能看到状态变化）
            current_step = getattr(env, '_sim_step_counter', 0) if hasattr(env, '_sim_step_counter') else 0
            should_print = (
                (current_step - env._last_prediction_method_print_step >= 500) or  # 从1000改为500，更频繁地打印
                (env._last_prediction_method != prediction_quality_good) or
                (env._last_prediction_method is None)  # 第一次也打印
            )
            
            if should_print:
                method_str = "神经网络预测" if prediction_quality_good else "线性外推"
                
                # 构建详细信息字符串
                detail_str = ""
                if quality_info is not None:
                    verified_status = quality_info.get('verified', False)
                    last_eval_info = quality_info.get('last_evaluation_info', None)
                    
                    if last_eval_info is not None:
                        # 有评估信息，显示详细原因
                        accurate_samples = last_eval_info.get('accurate_predictions', 0)
                        total_samples = last_eval_info.get('total_samples', 0)
                        required_ratio = last_eval_info.get('required_accuracy_ratio', 0.95)
                        accuracy_ratio = last_eval_info.get('accuracy_ratio', 0.0)
                        mean_error = last_eval_info.get('mean_error', float('inf'))
                        threshold = last_eval_info.get('threshold', 0.0)
                        
                        if not prediction_quality_good:
                            # 未通过评估，显示详细原因
                            detail_str = (
                                f"原因: 只有 {accurate_samples}/{total_samples} 个样本达到误差要求 "
                                f"(需要 {required_ratio:.1%}, 实际 {accuracy_ratio:.1%}), "
                                f"平均误差={mean_error:.4f} rad (阈值={threshold:.4f} rad)"
                            )
                        else:
                            # 通过评估，显示成功信息
                            detail_str = (
                                f"评估通过: {accurate_samples}/{total_samples} 个样本达到误差要求 "
                                f"(准确率 {accuracy_ratio:.1%} >= {required_ratio:.1%}), "
                                f"平均误差={mean_error:.4f} rad"
                            )
                    else:
                        # 没有评估信息，但已验证
                        if verified_status:
                            detail_str = "已验证通过（使用之前的评估结果）"
                        else:
                            detail_str = "未验证（尚未进行评估）"
                else:
                    # 预测器未初始化
                    detail_str = "预测器未初始化"
                
                print(f"[平台预测] 步骤 {current_step}: 使用 {method_str} - {detail_str}")
                env._last_prediction_method_print_step = current_step
                env._last_prediction_method = prediction_quality_good
            
            if prediction_quality_good:
                # 预测质量足够好：优先使用从机器狗观测预测的未来状态（关键改进）
                # 这样可以更好地捕捉周期性运动（如正弦运动），而不是简单的线性外推
                # 关键：使用从机器狗观测历史预测的平台运动，而不是平台历史数据
                dt = getattr(env, 'step_dt', 0.02)
                prediction_steps = max(1, int(prediction_time / dt))  # 计算需要预测多少步
                
                # 优先使用从机器狗观测预测的平台运动
                future_prediction_from_obs = env.get_platform_prediction_from_observations(prediction_steps=prediction_steps)
                
                if future_prediction_from_obs is not None:
                    # 使用从机器狗观测预测的未来状态（关键改进）
                    future_platform_roll = future_prediction_from_obs['roll']
                    future_platform_pitch = future_prediction_from_obs['pitch']
                    
                    # 检查预测是否合理（不是全零）
                    prediction_valid = (
                        (torch.abs(future_platform_roll) > 1e-6) | 
                        (torch.abs(future_platform_pitch) > 1e-6)
                    )
                    
                    # 计算未来误差：机器狗当前姿态 vs 未来平台姿态
                    future_orientation_error = torch.sqrt(
                        (future_platform_roll - robot_roll)**2 + 
                        (future_platform_pitch - robot_pitch)**2 + 1e-8
                    )
                    
                    # 如果预测有效，使用预测的未来误差；否则回退到改进的外推方法
                    if prediction_valid.any():
                        # 使用从机器狗观测预测的未来误差（对于有效的环境）
                        # 对于无效的环境，使用改进的外推方法作为后备
                        linear_future_roll, linear_future_pitch = env.extrapolate_platform_future_advanced(
                            prediction_time=prediction_time,
                            history_window=15  # 增加历史窗口以提高外推精度（从10增加到15）
                        )
                        linear_future_error = torch.sqrt(
                            (linear_future_roll - robot_roll)**2 + 
                            (linear_future_pitch - robot_pitch)**2 + 1e-8
                        )
                        
                        # 根据预测有效性选择使用神经网络预测还是改进的外推方法
                        future_orientation_error = torch.where(
                            prediction_valid,
                            future_orientation_error,  # 使用从机器狗观测预测
                            linear_future_error  # 使用改进的外推方法作为后备
                        )
                    else:
                        # 如果所有环境的预测都无效，使用改进的外推方法作为后备
                        future_platform_roll, future_platform_pitch = env.extrapolate_platform_future_advanced(
                            prediction_time=prediction_time,
                            history_window=15  # 统一使用15个历史窗口
                        )
                        future_orientation_error = torch.sqrt(
                            (future_platform_roll - robot_roll)**2 + 
                            (future_platform_pitch - robot_pitch)**2 + 1e-8
                        )
                else:
                    # 如果从机器狗观测预测不可用，尝试使用平台历史数据预测（后备方案）
                    future_prediction = env.get_platform_future_prediction(prediction_time=prediction_time)
                    
                    if future_prediction is not None:
                        # 使用平台历史数据预测的未来状态
                        future_platform_roll = future_prediction['roll']
                        future_platform_pitch = future_prediction['pitch']
                        
                        # 检查预测是否合理（不是全零）
                        prediction_valid = (
                            (torch.abs(future_platform_roll) > 1e-6) | 
                            (torch.abs(future_platform_pitch) > 1e-6)
                        )
                        
                        if prediction_valid.any():
                            # 计算未来误差
                            future_orientation_error = torch.sqrt(
                                (future_platform_roll - robot_roll)**2 + 
                                (future_platform_pitch - robot_pitch)**2 + 1e-8
                            )
                        else:
                            # 如果预测无效，使用改进的外推方法作为后备
                            future_platform_roll, future_platform_pitch = env.extrapolate_platform_future_advanced(
                                prediction_time=prediction_time,
                                history_window=15
                            )
                            future_orientation_error = torch.sqrt(
                                (future_platform_roll - robot_roll)**2 + 
                                (future_platform_pitch - robot_pitch)**2 + 1e-8
                            )
                    else:
                        # 如果预测不可用，使用改进的外推方法作为后备
                        future_platform_roll, future_platform_pitch = env.extrapolate_platform_future_advanced(
                            prediction_time=prediction_time,
                            history_window=15
                        )
                        future_orientation_error = torch.sqrt(
                            (future_platform_roll - robot_roll)**2 + 
                            (future_platform_pitch - robot_pitch)**2 + 1e-8
                        )
            else:
                # 预测质量不够好：使用改进的外推方法（基于历史数据和物理规律）
                # 使用二阶外推：θ(t+dt) = θ(t) + ω(t)*dt + 0.5*α(t)*dt²
                # 其中角加速度α从历史角速度数据中估计，更符合物理规律
                future_platform_roll, future_platform_pitch = env.extrapolate_platform_future_advanced(
                    prediction_time=prediction_time,
                    history_window=5  # 使用最近5个时间步估计角加速度
                )
                future_orientation_error = torch.sqrt(
                    (future_platform_roll - robot_roll)**2 + 
                    (future_platform_pitch - robot_pitch)**2 + 1e-8
                )
        
        # ========== 改进的误差组合策略 ==========
        # 目标：让机器狗更好地利用预测信息，提前响应平台运动
        
        # 改进1：根据预测质量动态调整权重
        # 如果预测质量好，更多权重给未来误差（鼓励提前响应）
        # 如果预测质量不够好，更多权重给当前误差（确保基础匹配）
        prediction_quality_good = env.is_platform_prediction_quality_good()
        
        if prediction_quality_good:
            # 预测质量好：80%未来误差 + 20%当前误差（更强调提前响应）
            future_weight = 0.8
            current_weight = 0.2
        else:
            # 预测质量不够好：60%未来误差 + 40%当前误差（平衡提前响应和基础匹配）
            future_weight = 0.6
            current_weight = 0.4
        
        orientation_error = future_weight * future_orientation_error + current_weight * current_orientation_error
        
        # 保存未来平台姿态作为目标姿态，用于方向正确性计算
        target_roll = future_platform_roll
        target_pitch = future_platform_pitch
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
            
            # 设置目标姿态：如果预测可用且合理，使用预测姿态；否则使用当前平台姿态
            target_roll = torch.where(use_prediction, predicted_platform_roll, current_platform_roll)
            target_pitch = torch.where(use_prediction, predicted_platform_pitch, current_platform_pitch)
        else:
            # 如果预测不可用，使用当前平台姿态（基础跟随）
            orientation_error = current_orientation_error
            target_roll = current_platform_roll
            target_pitch = current_platform_pitch
    
    # ========== 关键改进：奖励"朝向目标姿态移动"而不是"角速度匹配" ==========
    # 核心思想：
    # 1. 当机器人和平台姿态不一致时，机器人**应该**有更大的角速度来快速调整
    # 2. 不应该要求角速度匹配，而应该奖励机器人**朝向目标姿态移动**
    # 3. 主要奖励姿态匹配（特别是未来姿态），让机器人学会提前移动到目标位置
    
    # 目标姿态已经在上面计算好了（target_roll和target_pitch）
    # 这里直接使用，不需要重新计算
    
    # 计算姿态误差向量（目标姿态 - 当前姿态）
    roll_error = target_roll - robot_roll  # [num_envs]
    pitch_error = target_pitch - robot_pitch  # [num_envs]
    
    # 获取机器人的角速度（roll和pitch方向）
    robot_ang_vel_roll = robot.data.root_ang_vel_w[:, 0]  # [num_envs]
    robot_ang_vel_pitch = robot.data.root_ang_vel_w[:, 1]  # [num_envs]
    
    # 计算"方向正确性"奖励：奖励机器人朝向目标姿态移动
    # 如果误差为正，角速度应该为正；如果误差为负，角速度应该为负
    # 使用点积来衡量方向一致性：error · ang_vel > 0 表示方向正确
    direction_correctness_roll = roll_error * robot_ang_vel_roll  # 正数表示方向正确
    direction_correctness_pitch = pitch_error * robot_ang_vel_pitch  # 正数表示方向正确
    
    # 归一化：将方向正确性转换为0-1之间的奖励
    # 使用tanh函数：当方向完全正确时（error和ang_vel同号且较大），奖励接近1
    # 当方向错误时（error和ang_vel异号），奖励接近-1，然后映射到0-1
    direction_reward_roll = (torch.tanh(direction_correctness_roll / 0.1) + 1.0) / 2.0  # [0, 1]
    direction_reward_pitch = (torch.tanh(direction_correctness_pitch / 0.1) + 1.0) / 2.0  # [0, 1]
    
    # 当误差很小时，不需要大的角速度，此时方向奖励应该接近1（表示已经到位）
    # 当误差较大时，需要大的角速度，此时方向奖励应该反映方向是否正确
    error_magnitude = torch.sqrt(roll_error**2 + pitch_error**2 + 1e-8)  # [num_envs]
    small_error_mask = error_magnitude < 0.05  # 误差很小，已经基本到位
    
    # 对于小误差，方向奖励设为1（表示已经到位，不需要大的角速度）
    # 对于大误差，使用计算的方向奖励（鼓励朝向目标移动）
    direction_reward_roll = torch.where(small_error_mask, torch.ones_like(direction_reward_roll), direction_reward_roll)
    direction_reward_pitch = torch.where(small_error_mask, torch.ones_like(direction_reward_pitch), direction_reward_pitch)
    
    # 综合方向奖励（roll和pitch的平均）
    direction_reward = 0.5 * direction_reward_roll + 0.5 * direction_reward_pitch
    
    # 注意：这个方向奖励的权重应该很小，主要奖励还是姿态匹配
    # 方向奖励只是辅助，帮助机器人在姿态不一致时知道应该朝哪个方向移动
    
    # ========== 姿态匹配奖励（改进版：在中等误差范围提供更强梯度） ==========
    # 问题分析：
    # 1. 当误差在0.3左右时，exp(-0.3/0.08) ≈ 0.023，梯度非常小，无法继续优化
    # 2. 分段奖励只在误差<0.06时才有，对于0.1-0.3范围的误差没有额外的梯度
    # 解决：使用分段奖励函数，在中等误差范围（0.1-0.3）也提供额外的梯度
    
    # 基础奖励：使用指数奖励，在小误差时提供平滑的梯度
    # 当error=0时，reward=1；当error=std时，reward≈0.37
    orientation_reward_base = torch.exp(-orientation_error / std_orientation)
    
    # ========== 关键改进：分段奖励，覆盖从极小误差到中等误差的范围 ==========
    # 策略：在不同误差范围使用不同的奖励强度，确保在中等误差时仍有足够的梯度
    
    # 1. 极小误差（< 0.02 rad，约1.1度）：非常精确，给予最高奖励
    very_precise_bonus = torch.where(
        orientation_error < 0.02,
        torch.ones_like(orientation_reward_base) * 0.5,
        torch.zeros_like(orientation_reward_base)
    )
    
    # 2. 小误差（0.02-0.04 rad，约1.1-2.3度）：精确，给予较高奖励
    precise_bonus = torch.where(
        (orientation_error >= 0.02) & (orientation_error < 0.04),
        torch.ones_like(orientation_reward_base) * 0.3,
        torch.zeros_like(orientation_reward_base)
    )
    
    # 3. 良好误差（0.04-0.06 rad，约2.3-3.4度）：良好，给予中等奖励
    good_bonus = torch.where(
        (orientation_error >= 0.04) & (orientation_error < 0.06),
        torch.ones_like(orientation_reward_base) * 0.1,
        torch.zeros_like(orientation_reward_base)
    )
    
    # 4. 中等误差（0.06-0.15 rad，约3.4-8.6度）：关键改进区域
    # 在这个范围，基础指数奖励的梯度已经很小，需要额外的奖励来提供梯度
    # 使用线性衰减的奖励，确保在0.06-0.15范围内仍有足够的梯度
    medium_error_mask = (orientation_error >= 0.06) & (orientation_error < 0.15)
    # 线性衰减：从0.06时的0.05奖励，到0.15时的0.0奖励
    # 公式：reward = 0.05 * (1.0 - (error - 0.06) / (0.15 - 0.06))
    medium_bonus_value = 0.05 * (1.0 - (orientation_error - 0.06) / (0.15 - 0.06))
    medium_bonus_value = torch.clamp(medium_bonus_value, min=0.0, max=0.05)
    medium_bonus = torch.where(
        medium_error_mask,
        medium_bonus_value,
        torch.zeros_like(orientation_reward_base)
    )
    
    # 5. 较大误差（0.15-0.5 rad，约8.6-28.6度）：进一步改进区域
    # 在这个范围，基础指数奖励的梯度已经非常小，需要额外的奖励来提供梯度
    large_error_mask = (orientation_error >= 0.15) & (orientation_error < 0.5)
    # 线性衰减：从0.15时的0.05奖励，到0.5时的0.0奖励
    # 公式：reward = 0.05 * (1.0 - (error - 0.15) / (0.5 - 0.15))
    large_bonus_value = 0.05 * (1.0 - (orientation_error - 0.15) / (0.5 - 0.15))
    large_bonus_value = torch.clamp(large_bonus_value, min=0.0, max=0.05)
    large_bonus = torch.where(
        large_error_mask,
        large_bonus_value,
        torch.zeros_like(orientation_reward_base)
    )
    
    # 6. 很大误差（0.5-1.0 rad，约28.6-57.3度）：关键改进区域
    # 机器狗当前姿态误差平均0.856 rad，在这个范围，需要提供足够的梯度
    very_large_error_mask = (orientation_error >= 0.5) & (orientation_error < 1.0)
    # 线性衰减：从0.5时的0.03奖励，到1.0时的0.0奖励
    # 公式：reward = 0.03 * (1.0 - (error - 0.5) / (1.0 - 0.5))
    very_large_bonus_value = 0.03 * (1.0 - (orientation_error - 0.5) / (1.0 - 0.5))
    very_large_bonus_value = torch.clamp(very_large_bonus_value, min=0.0, max=0.03)
    very_large_bonus = torch.where(
        very_large_error_mask,
        very_large_bonus_value,
        torch.zeros_like(orientation_reward_base)
    )
    
    # 组合奖励：基础奖励 + 所有分段奖励
    # 这样在中等误差范围（0.1-1.0）时，除了基础指数奖励，还有额外的线性奖励提供梯度
    orientation_reward = (
        orientation_reward_base + 
        very_precise_bonus + 
        precise_bonus + 
        good_bonus + 
        medium_bonus + 
        large_bonus +
        very_large_bonus
    )
    orientation_reward = torch.clamp(orientation_reward, min=0.0, max=2.0)  # 最大奖励2.0
    
    # ========== 关键改进：运动中的跟随奖励更高 ==========
    # 问题：当平台跟随奖励权重过高时，机器人会过度关注姿态匹配，导致被动跟随，无法主动运动
    # 解决：让平台跟随奖励在机器人运动时给予更高的奖励（运动中的跟随比静止的跟随更有价值）
    
    # 获取速度命令（机器人的目标运动速度）
    try:
        velocity_commands = env.command_manager.get_command("base_velocity")  # [num_envs, 3] 或 [num_envs, ...]
        if velocity_commands.dim() == 1:
            velocity_commands = velocity_commands.unsqueeze(-1)  # [num_envs, 1]
        
        # 计算速度命令的大小（xy平面）
        if velocity_commands.shape[1] >= 2:
            command_vel_xy = torch.norm(velocity_commands[:, :2], dim=1)  # [num_envs]
        else:
            command_vel_xy = torch.abs(velocity_commands[:, 0])  # [num_envs]
        
        # 获取机器人实际速度（xy平面）
        robot_lin_vel_xy = torch.norm(robot.data.root_lin_vel_b[:, :2], dim=1)  # [num_envs]
        
        # 计算速度跟踪误差（越小越好）
        vel_tracking_error = torch.abs(command_vel_xy - robot_lin_vel_xy)
        
        # 运动奖励系数：当机器人执行速度命令时（运动），给予更高的奖励系数
        # 当速度跟踪误差小（正在运动）时，奖励系数接近1.5；当速度跟踪误差大（静止）时，奖励系数接近0.5
        # 这样运动中的跟随会获得更高的奖励，静止的跟随奖励较低
        motion_bonus_factor = 0.5 + 1.0 * torch.exp(-vel_tracking_error / 0.3)  # [0.5, 1.5]
        
        # 如果有速度命令，使用运动奖励系数；如果没有速度命令，使用较低的系数
        has_command = command_vel_xy > 0.01  # 有速度命令（阈值0.01 m/s）
        motion_bonus_factor = torch.where(
            has_command,
            motion_bonus_factor,  # 有速度命令时，根据速度跟踪误差给予奖励系数
            torch.ones_like(motion_bonus_factor) * 0.3  # 没有速度命令时，使用较低的系数（0.3）
        )
    except:
        # 如果无法获取速度命令，默认使用中等系数（1.0）
        motion_bonus_factor = torch.ones(env.num_envs, device=robot.device)
    
    # ========== 组合奖励：姿态匹配（主要）+ 方向正确性（辅助），运动时给予奖励加成 ==========
    # 权重分配：
    # - 姿态匹配：90%（主要奖励，强调精确的姿态匹配和提前响应）
    # - 方向正确性：10%（辅助奖励，帮助机器人在姿态不一致时知道应该朝哪个方向移动）
    # 
    # 关键改进：
    # 1. 保持原有的奖励结构（姿态匹配90% + 方向正确性10%）
    # 2. 添加运动奖励系数：运动中的跟随奖励更高（系数1.5），静止的跟随奖励较低（系数0.3）
    # 3. 这样机器人会学会在运动时跟随平台，而不是静止时跟随平台
    combined_reward = (0.9 * orientation_reward + 0.1 * direction_reward) * motion_bonus_factor
    
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


# ============================================================================
# 能量消耗相关奖励项
# ============================================================================

class power_consumption(ManagerTermBase):
    """惩罚关节功率消耗（扭矩 × 速度）
    
    用于降低机器狗的能量消耗，鼓励更高效的运动。
    功率 = |扭矩 × 角速度|
    """
    
    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        # 初始化基类
        super().__init__(cfg, env)
        # 获取默认参数
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset: Articulation = env.scene[asset_cfg.name]
        
        # 解析齿轮比（如果有的话，否则使用默认值1.0）
        self.gear_ratio = torch.ones(env.num_envs, asset.num_joints, device=env.device)
        if "gear_ratio" in cfg.params:
            index_list, _, value_list = string_utils.resolve_matching_names_values(
                cfg.params["gear_ratio"], asset.joint_names
            )
            self.gear_ratio[:, index_list] = torch.tensor(value_list, device=env.device)
        self.gear_ratio_scaled = self.gear_ratio / torch.max(self.gear_ratio)
    
    def __call__(
        self, 
        env: ManagerBasedRLEnv, 
        gear_ratio: dict[str, float] | None = None, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
        """计算功率消耗
        
        Args:
            env: 环境实例
            gear_ratio: 齿轮比字典（可选，如果提供会覆盖初始化时的设置）
            asset_cfg: 资产配置
        
        Returns:
            功率消耗 [num_envs]，单位：N·m·rad/s
        """
        asset: Articulation = env.scene[asset_cfg.name]
        
        # 获取关节扭矩和速度
        # 注意：对于PD控制器，action是位置目标，需要获取实际应用的扭矩
        joint_torque = asset.data.applied_torque[:, asset_cfg.joint_ids]  # [num_envs, num_joints]
        joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]  # [num_envs, num_joints]
        
        # 获取对应的齿轮比（处理slice和list两种情况）
        if isinstance(asset_cfg.joint_ids, slice):
            gear_ratio = self.gear_ratio_scaled[:, asset_cfg.joint_ids]
        else:
            gear_ratio = self.gear_ratio_scaled[:, asset_cfg.joint_ids]
        
        # 计算功率：|扭矩 × 角速度|（绝对值，因为功率总是正的）
        power = torch.abs(joint_torque * joint_vel * gear_ratio)
        
        # 返回总功率（所有关节的功率之和）
        return torch.sum(power, dim=-1)  # [num_envs]
