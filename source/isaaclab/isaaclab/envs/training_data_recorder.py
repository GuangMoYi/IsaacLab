# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Training data recorder for tracking platform prediction errors, base error ratios, and energy consumption."""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from scipy.stats import pearsonr
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class TrainingDataRecorder:
    """Records training metrics during simulation."""
    
    def __init__(self, env: ManagerBasedRLEnv, save_dir: str = "/home/user/IsaacLab/training_data"):
        """Initialize the training data recorder.
        
        Args:
            env: The environment instance.
            save_dir: Directory to save training data files.
        """
        self.env = env
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 记录间隔：每0.1秒记录一次
        self.record_interval = 0.1  # 秒
        # 保存间隔：每10秒保存一个文件
        self.save_interval = 10.0  # 秒
        
        # 计算记录间隔对应的步数
        self.record_interval_steps = int(self.record_interval / env.step_dt)
        self.save_interval_steps = int(self.save_interval / env.step_dt)
        
        # 时间序列数据缓冲区
        self.time_series_data = {
            'time': [],
            'prediction_error': [],  # 平台预测误差（瞬时值）
            'prediction_rmse': [],   # 平台预测误差RMSE（基于历史）
            'control_error': [],      # 姿态误差（瞬时值）
            'control_rmse': [],      # 姿态误差RMSE（基于历史）
            'base_error_ratio': [],   # 基座误差比值
            'energy_consumption': [], # 能量消耗
            'reward': [],             # 强化学习奖励（瞬时值）
            # 平台6自由度
            'platform_x': [], 'platform_y': [], 'platform_z': [],
            'platform_roll': [], 'platform_pitch': [], 'platform_yaw': [],
            # 机器狗基座6自由度
            'robot_x': [], 'robot_y': [], 'robot_z': [],
            'robot_roll': [], 'robot_pitch': [], 'robot_yaw': [],
            # 预测的平台6自由度
            'predicted_platform_x': [], 'predicted_platform_y': [], 'predicted_platform_z': [],
            'predicted_platform_roll': [], 'predicted_platform_pitch': [], 'predicted_platform_yaw': [],
        }
        
        # 单个数据记录
        self.single_data = {
            'prediction_errors': [],  # 预测器姿态误差历史
            'control_errors': [],  # 最终控制姿态误差历史
            'episode_lengths': [],  # 各环境的episode长度（存活时间）
        }
        
        # 能量消耗累计
        self.total_energy = 0.0
        self.total_time = 0.0
        
        # 当前文件索引
        self.current_file_index = 0
        
        # 步数计数器
        self.step_counter = 0
        
        # 删除旧文件
        self._cleanup_old_files()
        # 清空comparison_data中的平台运动数据
        self._cleanup_platform_motion_data()
    
    def _cleanup_old_files(self):
        """删除上一次训练保存的文件。"""
        import glob
        
        # 删除时间序列数据文件
        pattern = os.path.join(self.save_dir, "time_series_*.npz")
        for file in glob.glob(pattern):
            try:
                os.remove(file)
                print(f"[数据记录] 删除旧文件: {file}")
            except Exception as e:
                print(f"[数据记录] 删除文件失败 {file}: {e}")
        
        # 删除统计数据文件
        stats_file = os.path.join(self.save_dir, "training_statistics.npz")
        if os.path.exists(stats_file):
            try:
                os.remove(stats_file)
                print(f"[数据记录] 删除旧文件: {stats_file}")
            except Exception as e:
                print(f"[数据记录] 删除文件失败 {stats_file}: {e}")
        
        # 删除图像文件
        for pattern in ["prediction_error.png", "base_error_ratio.png", "energy_consumption.png", "training_statistics.png"]:
            img_file = os.path.join(self.save_dir, pattern)
            if os.path.exists(img_file):
                try:
                    os.remove(img_file)
                    print(f"[数据记录] 删除旧文件: {img_file}")
                except Exception as e:
                    print(f"[数据记录] 删除文件失败 {img_file}: {e}")
    
    def _cleanup_platform_motion_data(self):
        """清空comparison_data目录中的所有数据文件（包括平台运动数据和对比数据）。
        
        此函数在训练开始时自动调用，确保每次训练前都清除旧数据，防止多次训练数据过多。
        """
        import glob
        comparison_data_dir = "/home/user/IsaacLab/comparison_data"
        
        print(f"[数据记录] 训练开始前：正在清空comparison_data目录...")
        
        if not os.path.exists(comparison_data_dir):
            print(f"[数据记录] comparison_data目录不存在，创建目录: {comparison_data_dir}")
            os.makedirs(comparison_data_dir, exist_ok=True)
            return
        
        # 清除所有.npz文件（包括平台运动数据和对比数据）
        pattern = os.path.join(comparison_data_dir, "*.npz")
        files = glob.glob(pattern)
        
        if files:
            deleted_count = 0
            for file in files:
                try:
                    os.remove(file)
                    deleted_count += 1
                except Exception as e:
                    print(f"[数据记录] 删除文件失败 {file}: {e}")
            print(f"[数据记录] 训练开始前：已清空comparison_data目录，删除了 {deleted_count} 个文件")
            print(f"[数据记录] 这确保了每次训练前都清除旧数据，防止多次训练数据过多")
        else:
            print(f"[数据记录] comparison_data目录为空，无需清空")
    
    def record_step(self):
        """在每一步记录数据。"""
        self.step_counter += 1
        
        # 每0.1秒记录一次
        if self.step_counter % self.record_interval_steps == 0:
            self._record_time_series_data()
        
        # 每10秒保存一次文件
        if self.step_counter % self.save_interval_steps == 0:
            self._save_time_series_data()
        
        # 记录预测误差和控制误差（用于计算相关系数）
        self._record_prediction_and_control_errors()
        
        # 更新能量消耗
        self._update_energy_consumption()
    
    def _record_time_series_data(self):
        """记录时间序列数据。"""
        current_time = self.step_counter * self.env.step_dt
        
        # 1. 平台运动预测器输出与真实值之间的误差（瞬时值）
        prediction_error = self._compute_prediction_error()
        
        # 2. 平台预测误差RMSE（基于历史预测误差）
        prediction_rmse = self._compute_prediction_rmse()
        
        # 3. 姿态误差（瞬时值，从single_data获取最新的控制误差）
        control_error = self.single_data['control_errors'][-1] if len(self.single_data['control_errors']) > 0 else 0.0
        
        # 4. 姿态误差RMSE（基于历史控制误差）
        control_rmse = self._compute_control_rmse()
        
        # 5. 基座误差比值
        base_error_ratio = self._compute_base_error_ratio()
        
        # 6. 平均能量消耗（总能量消耗 / 总时间）
        avg_energy = self.total_energy / (self.total_time + 1e-8)
        
        # 7. 平台和机器狗的6自由度数据
        platform_6dof, robot_6dof = self._get_6dof_data()
        
        # 8. 强化学习奖励
        reward = self._get_reward()
        
        # 9. 预测的平台6自由度数据
        predicted_platform_6dof = self._get_predicted_platform_6dof()
        
        # 记录数据
        self.time_series_data['time'].append(current_time)
        self.time_series_data['prediction_error'].append(prediction_error)
        self.time_series_data['prediction_rmse'].append(prediction_rmse)
        self.time_series_data['control_error'].append(control_error)
        self.time_series_data['control_rmse'].append(control_rmse)
        self.time_series_data['base_error_ratio'].append(base_error_ratio)
        self.time_series_data['energy_consumption'].append(avg_energy)
        self.time_series_data['reward'].append(reward)
        # 平台6自由度
        self.time_series_data['platform_x'].append(platform_6dof[0])
        self.time_series_data['platform_y'].append(platform_6dof[1])
        self.time_series_data['platform_z'].append(platform_6dof[2])
        self.time_series_data['platform_roll'].append(platform_6dof[3])
        self.time_series_data['platform_pitch'].append(platform_6dof[4])
        self.time_series_data['platform_yaw'].append(platform_6dof[5])
        # 机器狗基座6自由度
        self.time_series_data['robot_x'].append(robot_6dof[0])
        self.time_series_data['robot_y'].append(robot_6dof[1])
        self.time_series_data['robot_z'].append(robot_6dof[2])
        self.time_series_data['robot_roll'].append(robot_6dof[3])
        self.time_series_data['robot_pitch'].append(robot_6dof[4])
        self.time_series_data['robot_yaw'].append(robot_6dof[5])
        # 预测的平台6自由度
        self.time_series_data['predicted_platform_x'].append(predicted_platform_6dof[0])
        self.time_series_data['predicted_platform_y'].append(predicted_platform_6dof[1])
        self.time_series_data['predicted_platform_z'].append(predicted_platform_6dof[2])
        self.time_series_data['predicted_platform_roll'].append(predicted_platform_6dof[3])
        self.time_series_data['predicted_platform_pitch'].append(predicted_platform_6dof[4])
        self.time_series_data['predicted_platform_yaw'].append(predicted_platform_6dof[5])
    
    def _compute_prediction_error(self) -> float:
        """计算平台运动预测器输出与真实值之间的平均误差。"""
        # 检查预测器是否可用
        if not hasattr(self.env, '_platform_predictor') or self.env._platform_predictor is None:
            return 0.0
        
        # 检查预测器是否已验证（如果未验证，说明预测器还没准备好）
        if hasattr(self.env._platform_predictor, 'prediction_quality_verified'):
            if not self.env._platform_predictor.prediction_quality_verified:
                return 0.0
        
        try:
            # 获取预测值
            platform_prediction = self.env.get_platform_prediction_for_observation(delay_steps=5)
            if platform_prediction is None:
                return 0.0
            
            predicted_roll = platform_prediction['roll']  # [num_envs]
            predicted_pitch = platform_prediction['pitch']  # [num_envs]
            
            # 获取真实值
            robot = self.env.scene["robot"]
            platform = self.env.scene["platform"]
            
            import isaaclab.utils.math as math_utils
            
            robot_roll, robot_pitch, _ = torch.stack(
                math_utils.euler_zyx_from_quat(robot.data.root_quat_w), dim=1
            ).split(1, dim=1)
            robot_roll = robot_roll.squeeze(1)
            robot_pitch = robot_pitch.squeeze(1)
            
            platform_roll, platform_pitch, _ = torch.stack(
                math_utils.euler_zyx_from_quat(platform.data.root_quat_w), dim=1
            ).split(1, dim=1)
            platform_roll = platform_roll.squeeze(1)
            platform_pitch = platform_pitch.squeeze(1)
            
            # 计算预测误差（预测的平台姿态 - 真实平台姿态）
            roll_error = predicted_roll - platform_roll
            pitch_error = predicted_pitch - platform_pitch
            
            # 计算平均误差（所有环境的平均）
            error = torch.sqrt(roll_error**2 + pitch_error**2 + 1e-8)
            mean_error = error.mean().item()
            
            return mean_error
        except Exception:
            # 静默失败，不打印错误（避免刷屏）
            return 0.0
    
    def _compute_base_error_ratio(self) -> float:
        """计算基座误差比值（各环境的平均）。"""
        try:
            from isaaclab.envs.mdp.observations import orientation_error_ratio_metric
            
            ratio = orientation_error_ratio_metric(self.env)
            mean_ratio = ratio.mean().item()
            
            return mean_ratio
        except Exception as e:
            print(f"[数据记录] 计算基座误差比值失败: {e}")
            return 0.0
    
    def _record_prediction_and_control_errors(self):
        """记录预测误差和控制误差（用于计算皮尔逊相关系数）。
        
        关键修复：确保预测误差和控制误差总是成对记录，即使预测器未验证。
        这样在计算相关系数时，两个列表的长度会保持一致。
        """
        import isaaclab.utils.math as math_utils
        robot = self.env.scene["robot"]
        platform = self.env.scene["platform"]
        
        # 获取真实平台姿态和机器狗姿态
        platform_roll, platform_pitch, _ = torch.stack(
            math_utils.euler_zyx_from_quat(platform.data.root_quat_w), dim=1
        ).split(1, dim=1)
        platform_roll = platform_roll.squeeze(1)
        platform_pitch = platform_pitch.squeeze(1)
        
        robot_roll, robot_pitch, _ = torch.stack(
            math_utils.euler_zyx_from_quat(robot.data.root_quat_w), dim=1
        ).split(1, dim=1)
        robot_roll = robot_roll.squeeze(1)
        robot_pitch = robot_pitch.squeeze(1)
        
        # 计算最终控制姿态误差（机器狗与平台的姿态误差）
        # 这个总是可以计算的，不依赖于预测器
        control_roll_error = robot_roll - platform_roll
        control_pitch_error = robot_pitch - platform_pitch
        control_error = torch.sqrt(control_roll_error**2 + control_pitch_error**2 + 1e-8)
        self.single_data['control_errors'].append(control_error.mean().item())
        
        # 计算预测器姿态误差（只有当预测器可用且已验证时才计算）
        prediction_error_recorded = False
        if hasattr(self.env, '_platform_predictor') and self.env._platform_predictor is not None:
            # 检查预测器是否已验证
            is_verified = True
            if hasattr(self.env._platform_predictor, 'prediction_quality_verified'):
                is_verified = self.env._platform_predictor.prediction_quality_verified
            
            if is_verified:
                try:
                    # 计算预测器姿态误差
                    platform_prediction = self.env.get_platform_prediction_for_observation(delay_steps=5)
                    if platform_prediction is not None:
                        predicted_roll = platform_prediction['roll']
                        predicted_pitch = platform_prediction['pitch']
                        
                        # 预测误差（预测的平台姿态 - 真实平台姿态）
                        pred_roll_error = predicted_roll - platform_roll
                        pred_pitch_error = predicted_pitch - platform_pitch
                        prediction_error = torch.sqrt(pred_roll_error**2 + pred_pitch_error**2 + 1e-8)
                        self.single_data['prediction_errors'].append(prediction_error.mean().item())
                        prediction_error_recorded = True
                except Exception as e:
                    # 预测误差计算失败
                    if not hasattr(self, '_prediction_error_fail_printed'):
                        print(f"[数据记录] 计算预测误差失败（后续将静默）: {e}")
                        self._prediction_error_fail_printed = True
        
        # 关键修复：如果预测误差未记录（预测器不可用或未验证），也记录一个占位值
        # 这样可以确保两个列表长度一致，但占位值不应该用于计算相关系数
        # 我们使用NaN作为占位值，在计算相关系数时会自动忽略
        if not prediction_error_recorded:
            self.single_data['prediction_errors'].append(float('nan'))
    
    def _get_reward(self) -> float:
        """获取强化学习奖励（所有环境的平均）。"""
        try:
            if hasattr(self.env, 'reward_buf'):
                reward_buf = self.env.reward_buf  # [num_envs]
                return reward_buf.mean().item()
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _get_predicted_platform_6dof(self) -> list:
        """获取预测的平台6自由度数据（相对于初始位置）。
        
        Returns:
            list: [x, y, z, roll, pitch, yaw] 预测的平台6自由度数据（所有环境的平均）
        """
        try:
            platform = self.env.scene["platform"]
            import isaaclab.utils.math as math_utils
            
            # 获取初始位置（用于计算相对位置）
            if not hasattr(self, '_initial_platform_pos'):
                self._initial_platform_pos = platform.data.root_pos_w.clone()
            
            # 获取真实平台位置和姿态（相对于初始位置）
            platform_pos = (platform.data.root_pos_w - self._initial_platform_pos).mean(dim=0).cpu().numpy()
            platform_x = float(platform_pos[0])
            platform_y = float(platform_pos[1])
            platform_z = float(platform_pos[2])
            
            # 获取真实平台姿态
            platform_euler = torch.stack(
                math_utils.euler_zyx_from_quat(platform.data.root_quat_w), dim=1
            ).mean(dim=0).cpu().numpy()
            platform_roll = float(platform_euler[0])
            platform_pitch = float(platform_euler[1])
            platform_yaw = float(platform_euler[2])
            
            # 默认使用真实值（如果预测器不可用）
            predicted_roll = platform_roll
            predicted_pitch = platform_pitch
            
            # 尝试获取预测器的预测值
            if hasattr(self.env, '_platform_predictor') and self.env._platform_predictor is not None:
                # 检查预测器是否已验证
                prediction_verified = True
                if hasattr(self.env._platform_predictor, 'prediction_quality_verified'):
                    prediction_verified = self.env._platform_predictor.prediction_quality_verified
                
                if prediction_verified:
                    # 获取预测的roll和pitch
                    platform_prediction = self.env.get_platform_prediction_for_observation(delay_steps=5)
                    if platform_prediction is not None:
                        predicted_roll_tensor = platform_prediction.get('roll', None)
                        predicted_pitch_tensor = platform_prediction.get('pitch', None)
                        
                        if predicted_roll_tensor is not None:
                            predicted_roll = float(predicted_roll_tensor.mean().item())
                        if predicted_pitch_tensor is not None:
                            predicted_pitch = float(predicted_pitch_tensor.mean().item())
            
            # 返回预测值（roll, pitch）和真实值（x, y, z, yaw）
            # 注意：如果预测器不可用，roll和pitch也使用真实值
            return [
                platform_x,  # 使用真实值（预测器不预测位置）
                platform_y,
                platform_z,
                predicted_roll,  # 使用预测值（如果可用），否则使用真实值
                predicted_pitch,  # 使用预测值（如果可用），否则使用真实值
                platform_yaw,  # 使用真实值（预测器不预测yaw）
            ]
        except Exception as e:
            # 如果获取失败，打印错误信息（仅第一次）
            if not hasattr(self, '_prediction_error_printed'):
                print(f"[数据记录] 获取预测平台6自由度数据失败: {e}")
                import traceback
                traceback.print_exc()
                self._prediction_error_printed = True
            # 返回真实值作为占位（而不是零值）
            try:
                platform = self.env.scene["platform"]
                import isaaclab.utils.math as math_utils
                if not hasattr(self, '_initial_platform_pos'):
                    self._initial_platform_pos = platform.data.root_pos_w.clone()
                platform_pos = (platform.data.root_pos_w - self._initial_platform_pos).mean(dim=0).cpu().numpy()
                platform_euler = torch.stack(
                    math_utils.euler_zyx_from_quat(platform.data.root_quat_w), dim=1
                ).mean(dim=0).cpu().numpy()
                return [
                    float(platform_pos[0]),
                    float(platform_pos[1]),
                    float(platform_pos[2]),
                    float(platform_euler[0]),
                    float(platform_euler[1]),
                    float(platform_euler[2]),
                ]
            except:
                return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    def _update_energy_consumption(self):
        """更新能量消耗累计。"""
        try:
            robot = self.env.scene["robot"]
            
            # 获取关节扭矩和速度
            joint_torque = robot.data.applied_torque  # [num_envs, num_joints]
            joint_vel = robot.data.joint_vel  # [num_envs, num_joints]
            
            # 计算功率：|τ_j * q̇_j|（绝对值）
            power = torch.abs(joint_torque * joint_vel)  # [num_envs, num_joints]
            total_power = power.sum(dim=1)  # [num_envs] 每个环境的总功率
            
            # 累计能量消耗（功率 * 时间步长）
            energy_step = total_power.mean().item() * self.env.step_dt
            self.total_energy += energy_step
            self.total_time += self.env.step_dt
            
        except Exception as e:
            print(f"[数据记录] 更新能量消耗失败: {e}")
    
    def _get_6dof_data(self) -> tuple[list, list]:
        """获取平台和机器狗的6自由度数据（各环境的平均）。
        
        Returns:
            (platform_6dof, robot_6dof): 每个都是[x, y, z, roll, pitch, yaw]
        """
        try:
            import isaaclab.utils.math as math_utils
            robot = self.env.scene["robot"]
            platform = self.env.scene["platform"]
            
            # 获取初始位置（用于计算相对位置）
            if not hasattr(self, '_initial_platform_pos'):
                self._initial_platform_pos = platform.data.root_pos_w.clone()
            if not hasattr(self, '_initial_robot_pos'):
                self._initial_robot_pos = robot.data.root_pos_w.clone()
            
            # 平台位置（相对位置）
            platform_pos = (platform.data.root_pos_w - self._initial_platform_pos).mean(dim=0).cpu().numpy()
            # 平台姿态（欧拉角）
            platform_euler = torch.stack(
                math_utils.euler_zyx_from_quat(platform.data.root_quat_w), dim=1
            ).mean(dim=0).cpu().numpy()
            platform_6dof = [
                float(platform_pos[0]), float(platform_pos[1]), float(platform_pos[2]),
                float(platform_euler[0]), float(platform_euler[1]), float(platform_euler[2])
            ]
            
            # 机器狗位置（相对位置）
            robot_pos = (robot.data.root_pos_w - self._initial_robot_pos).mean(dim=0).cpu().numpy()
            # 机器狗姿态（欧拉角）
            robot_euler = torch.stack(
                math_utils.euler_zyx_from_quat(robot.data.root_quat_w), dim=1
            ).mean(dim=0).cpu().numpy()
            robot_6dof = [
                float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2]),
                float(robot_euler[0]), float(robot_euler[1]), float(robot_euler[2])
            ]
            
            return platform_6dof, robot_6dof
        except Exception:
            return [0.0] * 6, [0.0] * 6
    
    def _compute_prediction_rmse(self) -> float:
        """计算平台预测误差RMSE（基于历史预测误差的均方根误差）。"""
        try:
            if len(self.single_data['prediction_errors']) == 0:
                return 0.0
            
            # 计算最近一段时间的RMSE（使用最近1000个数据点，如果不足则使用全部）
            errors = self.single_data['prediction_errors']
            recent_errors = errors[-1000:] if len(errors) > 1000 else errors
            
            # 计算RMSE
            rmse = np.sqrt(np.mean(np.array(recent_errors)**2))
            return rmse
        except Exception:
            return 0.0
    
    def _compute_control_rmse(self) -> float:
        """计算姿态误差RMSE（基于历史控制误差的均方根误差）。"""
        try:
            if len(self.single_data['control_errors']) == 0:
                return 0.0
            
            # 计算最近一段时间的RMSE（使用最近1000个数据点，如果不足则使用全部）
            errors = self.single_data['control_errors']
            recent_errors = errors[-1000:] if len(errors) > 1000 else errors
            
            # 计算RMSE
            rmse = np.sqrt(np.mean(np.array(recent_errors)**2))
            return rmse
        except Exception:
            return 0.0
    
    def _save_time_series_data(self):
        """保存时间序列数据到文件。"""
        if len(self.time_series_data['time']) == 0:
            return
        
        filename = os.path.join(self.save_dir, f"time_series_{self.current_file_index:04d}.npz")
        
        # 转换为numpy数组
        data = {
            'time': np.array(self.time_series_data['time']),
            'prediction_error': np.array(self.time_series_data['prediction_error']),
            'prediction_rmse': np.array(self.time_series_data['prediction_rmse']),
            'control_error': np.array(self.time_series_data['control_error']),
            'control_rmse': np.array(self.time_series_data['control_rmse']),
            'base_error_ratio': np.array(self.time_series_data['base_error_ratio']),
            'energy_consumption': np.array(self.time_series_data['energy_consumption']),
            'reward': np.array(self.time_series_data['reward']),
            # 平台6自由度
            'platform_x': np.array(self.time_series_data['platform_x']),
            'platform_y': np.array(self.time_series_data['platform_y']),
            'platform_z': np.array(self.time_series_data['platform_z']),
            'platform_roll': np.array(self.time_series_data['platform_roll']),
            'platform_pitch': np.array(self.time_series_data['platform_pitch']),
            'platform_yaw': np.array(self.time_series_data['platform_yaw']),
            # 机器狗基座6自由度
            'robot_x': np.array(self.time_series_data['robot_x']),
            'robot_y': np.array(self.time_series_data['robot_y']),
            'robot_z': np.array(self.time_series_data['robot_z']),
            'robot_roll': np.array(self.time_series_data['robot_roll']),
            'robot_pitch': np.array(self.time_series_data['robot_pitch']),
            'robot_yaw': np.array(self.time_series_data['robot_yaw']),
            # 预测的平台6自由度
            'predicted_platform_x': np.array(self.time_series_data['predicted_platform_x']),
            'predicted_platform_y': np.array(self.time_series_data['predicted_platform_y']),
            'predicted_platform_z': np.array(self.time_series_data['predicted_platform_z']),
            'predicted_platform_roll': np.array(self.time_series_data['predicted_platform_roll']),
            'predicted_platform_pitch': np.array(self.time_series_data['predicted_platform_pitch']),
            'predicted_platform_yaw': np.array(self.time_series_data['predicted_platform_yaw']),
        }
        
        # 同时保存为CSV格式（便于查看和画图）
        csv_filename = filename.replace('.npz', '.csv')
        df = pd.DataFrame(data)
        df.to_csv(csv_filename, index=False)
        
        np.savez(filename, **data)
        print(f"[数据记录] 保存时间序列数据到: {filename}")
        
        # 清空缓冲区
        self.time_series_data = {key: [] for key in self.time_series_data}
        
        # 更新文件索引
        self.current_file_index += 1
    
    def record_episode_end(self, env_ids: torch.Tensor):
        """记录episode结束时的数据（存活时间）。"""
        try:
            # 记录各环境的episode长度（存活时间）
            episode_lengths = self.env.episode_length_buf[env_ids].cpu().numpy()
            self.single_data['episode_lengths'].extend(episode_lengths.tolist())
        except Exception as e:
            print(f"[数据记录] 记录episode结束数据失败: {e}")
    
    def save_final_statistics(self):
        """保存最终统计数据。"""
        # 保存剩余的时间序列数据
        if len(self.time_series_data['time']) > 0:
            self._save_time_series_data()
        
        # 计算皮尔逊相关系数
        correlation = self._compute_pearson_correlation()
        
        # 计算平均存活时间
        avg_survival_time = self._compute_avg_survival_time()
        
        # 保存统计数据
        stats_file = os.path.join(self.save_dir, "training_statistics.npz")
        stats_data = {
            'prediction_control_correlation': correlation,
            'avg_survival_time': avg_survival_time,
            'total_energy': self.total_energy,
            'total_time': self.total_time,
            'avg_energy_consumption': self.total_energy / (self.total_time + 1e-8),
        }
        
        np.savez(stats_file, **stats_data)
        print(f"[数据记录] 保存统计数据到: {stats_file}")
        print(f"[数据记录] 预测器与控制误差相关系数: {correlation:.4f}")
        print(f"[数据记录] 平均存活时间: {avg_survival_time:.2f} 步")
        print(f"[数据记录] 平均能量消耗: {stats_data['avg_energy_consumption']:.4f}")
    
    def _compute_pearson_correlation(self) -> float:
        """计算预测器误差与控制误差的皮尔逊相关系数。
        
        关键修复：只使用有效的预测误差数据（非NaN），确保两个列表长度一致。
        """
        try:
            pred_len = len(self.single_data['prediction_errors'])
            control_len = len(self.single_data['control_errors'])
            
            # 调试信息：打印数据长度
            print(f"[数据记录] 计算相关系数 - 预测误差数量: {pred_len}, 控制误差数量: {control_len}")
            
            if pred_len < 2 or control_len < 2:
                print(f"[数据记录] 数据不足，无法计算相关系数（需要至少2个数据点）")
                return 0.0
            
            prediction_errors = np.array(self.single_data['prediction_errors'])
            control_errors = np.array(self.single_data['control_errors'])
            
            # 关键修复：只使用有效的预测误差数据（非NaN）
            # 同时确保对应的控制误差也是有效的
            valid_mask = ~np.isnan(prediction_errors) & ~np.isnan(control_errors)
            prediction_errors_valid = prediction_errors[valid_mask]
            control_errors_valid = control_errors[valid_mask]
            
            valid_count = valid_mask.sum()
            print(f"[数据记录] 有效数据点数量: {valid_count} / {len(prediction_errors)} (NaN数量: {(~valid_mask).sum()})")
            
            if valid_count < 2:
                print(f"[数据记录] 有效数据不足，无法计算相关系数（需要至少2个有效数据点）")
                return 0.0
            
            # 检查数据是否有变化（不是常数）
            pred_std = prediction_errors_valid.std()
            control_std = control_errors_valid.std()
            
            print(f"[数据记录] 预测误差统计 - 均值: {prediction_errors_valid.mean():.6f}, 标准差: {pred_std:.6f}, 范围: [{prediction_errors_valid.min():.6f}, {prediction_errors_valid.max():.6f}]")
            print(f"[数据记录] 控制误差统计 - 均值: {control_errors_valid.mean():.6f}, 标准差: {control_std:.6f}, 范围: [{control_errors_valid.min():.6f}, {control_errors_valid.max():.6f}]")
            
            if pred_std < 1e-10:
                print(f"[数据记录] 预测误差是常数（std={pred_std:.2e}），无法计算相关系数")
                return 0.0
            
            if control_std < 1e-10:
                print(f"[数据记录] 控制误差是常数（std={control_std:.2e}），无法计算相关系数")
                return 0.0
            
            # 计算相关系数（只使用有效数据）
            correlation, p_value = pearsonr(prediction_errors_valid, control_errors_valid)
            
            if np.isnan(correlation) or np.isinf(correlation):
                print(f"[数据记录] 相关系数为NaN或Inf，返回0.0")
                return 0.0
            
            print(f"[数据记录] 皮尔逊相关系数: {correlation:.6f}, p值: {p_value:.6f}")
            return correlation
        except Exception as e:
            print(f"[数据记录] 计算皮尔逊相关系数失败: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def _compute_avg_survival_time(self) -> float:
        """计算平均存活时间（步数）。"""
        try:
            if len(self.single_data['episode_lengths']) == 0:
                return 0.0
            return np.mean(self.single_data['episode_lengths'])
        except Exception as e:
            print(f"[数据记录] 计算平均存活时间失败: {e}")
            return 0.0
    

