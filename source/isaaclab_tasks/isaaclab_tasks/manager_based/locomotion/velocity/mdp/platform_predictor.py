# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""神经网络平台运动预测器

使用神经网络预测平台未来运动，支持在线学习。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PlatformMotionPredictor(nn.Module):
    """神经网络平台运动预测器
    
    输入：历史平台姿态（roll, pitch）和角速度（roll, pitch）
    输出：预测的未来姿态（roll, pitch）和角速度（roll, pitch）
    
    支持在线学习：使用实际平台状态更新网络参数
    """
    
    def __init__(
        self,
        history_length: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        prediction_horizon: float = 0.2,
        learning_rate: float = 1e-3,
        device: str = "cuda",
    ):
        """初始化预测器
        
        Args:
            history_length: 使用的历史长度（步数）
            hidden_size: 隐藏层大小
            num_layers: LSTM层数
            prediction_horizon: 预测时间范围（秒）
            learning_rate: 学习率
            device: 设备（cuda/cpu）
        """
        super().__init__()
        
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.device = device
        
        # 输入维度：历史姿态（roll, pitch）和角速度（roll, pitch）
        # 每个时间步4个特征，历史长度history_length
        input_size = 4  # [roll, pitch, roll_ang_vel, pitch_ang_vel]
        
        # LSTM层：处理时序数据
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        
        # 输出层：预测未来姿态和角速度
        # 输出：未来姿态（roll, pitch）和角速度（roll, pitch）
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 4)  # [predicted_roll, predicted_pitch, predicted_roll_ang_vel, predicted_pitch_ang_vel]
        )
        
        # 优化器
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 训练历史（用于批量更新）
        self.training_buffer = {
            'inputs': [],
            'targets': [],
            'max_buffer_size': 200,  # 增加缓冲区大小：从100增加到200，保留更多训练样本
        }
        
        # 训练计数器
        self.train_step_count = 0
        self.batch_size = 64  # 增加批量大小：从32增加到64，更稳定的梯度估计
        
    def forward(self, history_roll: torch.Tensor, history_pitch: torch.Tensor,
                history_roll_ang_vel: torch.Tensor, history_pitch_ang_vel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """前向传播：预测未来姿态和角速度
        
        Args:
            history_roll: 历史roll角度 [batch_size, history_length]
            history_pitch: 历史pitch角度 [batch_size, history_length]
            history_roll_ang_vel: 历史roll角速度 [batch_size, history_length]
            history_pitch_ang_vel: 历史pitch角速度 [batch_size, history_length]
        
        Returns:
            predicted_roll: 预测的roll角度 [batch_size]
            predicted_pitch: 预测的pitch角度 [batch_size]
            predicted_roll_ang_vel: 预测的roll角速度 [batch_size]
            predicted_pitch_ang_vel: 预测的pitch角速度 [batch_size]
        """
        batch_size = history_roll.shape[0]
        
        # 组合输入：[batch_size, history_length, 4]
        # 注意：即使输入数据被detach，stack操作仍然可以正常进行
        # 但我们需要确保输入数据在正确的设备上
        inputs = torch.stack([
            history_roll,
            history_pitch,
            history_roll_ang_vel,
            history_pitch_ang_vel
        ], dim=-1)  # [batch_size, history_length, 4]
        
        # 确保inputs在正确的设备上
        inputs = inputs.to(next(self.parameters()).device)
        
        # LSTM处理
        lstm_out, _ = self.lstm(inputs)  # [batch_size, history_length, hidden_size]
        
        # 使用最后一个时间步的输出
        last_hidden = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        
        # 全连接层输出
        output = self.fc(last_hidden)  # [batch_size, 4]
        
        # 分离输出
        predicted_roll = output[:, 0]
        predicted_pitch = output[:, 1]
        predicted_roll_ang_vel = output[:, 2]
        predicted_pitch_ang_vel = output[:, 3]
        
        return predicted_roll, predicted_pitch, predicted_roll_ang_vel, predicted_pitch_ang_vel
    
    def predict(self, platform_history: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """预测平台未来状态
        
        Args:
            platform_history: 平台历史数据字典，包含：
                - 'quat_w': 历史四元数（可以是tensor stack或列表）[history_length, num_envs, 4]
                - 'ang_vel_w': 历史角速度（可以是tensor stack或列表）[history_length, num_envs, 3]
        
        Returns:
            predicted_roll: 预测的roll角度 [num_envs]
            predicted_pitch: 预测的pitch角度 [num_envs]
            predicted_roll_ang_vel: 预测的roll角速度 [num_envs]
            predicted_pitch_ang_vel: 预测的pitch角速度 [num_envs]
        """
        from isaaclab.utils.math import euler_xyz_from_quat
        
        if platform_history.get('quat_w') is None or platform_history.get('ang_vel_w') is None:
            # 如果没有历史数据，返回零
            num_envs = 1  # 默认值
            if platform_history.get('current_ang_vel_w') is not None:
                num_envs = platform_history['current_ang_vel_w'].shape[0]
            device = next(self.parameters()).device
            return (
                torch.zeros(num_envs, device=device),
                torch.zeros(num_envs, device=device),
                torch.zeros(num_envs, device=device),
                torch.zeros(num_envs, device=device)
            )
        
        quat_history = platform_history['quat_w']
        ang_vel_history = platform_history['ang_vel_w']
        
        # 处理列表或tensor格式
        if isinstance(quat_history, list):
            quat_history = torch.stack(quat_history, dim=0)  # [history_length, num_envs, 4]
        if isinstance(ang_vel_history, list):
            ang_vel_history = torch.stack(ang_vel_history, dim=0)  # [history_length, num_envs, 3]
        
        # 限制历史长度
        actual_history_length = min(self.history_length, quat_history.shape[0])
        if actual_history_length < 2:
            # 历史数据不足，返回零
            num_envs = quat_history.shape[1] if quat_history.shape[0] > 0 else 1
            device = next(self.parameters()).device
            return (
                torch.zeros(num_envs, device=device),
                torch.zeros(num_envs, device=device),
                torch.zeros(num_envs, device=device),
                torch.zeros(num_envs, device=device)
            )
        
        # 提取历史数据
        recent_quat = quat_history[-actual_history_length:]  # [actual_history_length, num_envs, 4]
        recent_ang_vel = ang_vel_history[-actual_history_length:]  # [actual_history_length, num_envs, 3]
        
        num_envs = recent_quat.shape[1]
        
        # 提取roll和pitch角度
        history_roll = torch.zeros(actual_history_length, num_envs, device=recent_quat.device)
        history_pitch = torch.zeros(actual_history_length, num_envs, device=recent_quat.device)
        
        for i in range(actual_history_length):
            roll_i, pitch_i, _ = euler_xyz_from_quat(recent_quat[i])
            history_roll[i] = roll_i
            history_pitch[i] = pitch_i
        
        # 提取roll和pitch角速度
        history_roll_ang_vel = recent_ang_vel[:, :, 0]  # [actual_history_length, num_envs]
        history_pitch_ang_vel = recent_ang_vel[:, :, 1]  # [actual_history_length, num_envs]
        
        # 转置为 [num_envs, actual_history_length]
        history_roll = history_roll.transpose(0, 1)
        history_pitch = history_pitch.transpose(0, 1)
        history_roll_ang_vel = history_roll_ang_vel.transpose(0, 1)
        history_pitch_ang_vel = history_pitch_ang_vel.transpose(0, 1)
        
        # 如果历史长度不足，用第一个值填充
        if actual_history_length < self.history_length:
            padding_size = self.history_length - actual_history_length
            history_roll = torch.cat([
                history_roll[:, 0:1].expand(-1, padding_size),
                history_roll
            ], dim=1)
            history_pitch = torch.cat([
                history_pitch[:, 0:1].expand(-1, padding_size),
                history_pitch
            ], dim=1)
            history_roll_ang_vel = torch.cat([
                history_roll_ang_vel[:, 0:1].expand(-1, padding_size),
                history_roll_ang_vel
            ], dim=1)
            history_pitch_ang_vel = torch.cat([
                history_pitch_ang_vel[:, 0:1].expand(-1, padding_size),
                history_pitch_ang_vel
            ], dim=1)
        
        # 预测
        self.eval()
        with torch.no_grad():
            predicted_roll, predicted_pitch, predicted_roll_ang_vel, predicted_pitch_ang_vel = self.forward(
                history_roll, history_pitch, history_roll_ang_vel, history_pitch_ang_vel
            )
        
        return predicted_roll, predicted_pitch, predicted_roll_ang_vel, predicted_pitch_ang_vel
    
    def predict_current_from_delayed_history(
        self, 
        platform_history: dict,
        delay_steps: int = 5
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """使用t-delay_steps之前的数据预测当前时刻（t时刻）的平台状态
        
        这个方法专门用于观测空间：机器狗能看到t-5之前的数据，然后预测当前时刻的平台状态
        
        Args:
            platform_history: 平台历史数据字典，包含：
                - 'quat_w': 历史四元数（可以是tensor stack或列表）[history_length, num_envs, 4]
                - 'ang_vel_w': 历史角速度（可以是tensor stack或列表）[history_length, num_envs, 3]
            delay_steps: 延迟步数，使用t-delay_steps之前的数据来预测t时刻
        
        Returns:
            predicted_roll: 预测的当前roll角度 [num_envs]
            predicted_pitch: 预测的当前pitch角度 [num_envs]
            predicted_roll_ang_vel: 预测的当前roll角速度 [num_envs]
            predicted_pitch_ang_vel: 预测的当前pitch角速度 [num_envs]
        """
        from isaaclab.utils.math import euler_xyz_from_quat
        
        if platform_history.get('quat_w') is None or platform_history.get('ang_vel_w') is None:
            # 如果没有历史数据，返回零
            num_envs = 1  # 默认值
            if platform_history.get('current_ang_vel_w') is not None:
                num_envs = platform_history['current_ang_vel_w'].shape[0]
            device = next(self.parameters()).device
            return (
                torch.zeros(num_envs, device=device),
                torch.zeros(num_envs, device=device),
                torch.zeros(num_envs, device=device),
                torch.zeros(num_envs, device=device)
            )
        
        quat_history = platform_history['quat_w']
        ang_vel_history = platform_history['ang_vel_w']
        
        # 处理列表或tensor格式
        if isinstance(quat_history, list):
            quat_history = torch.stack(quat_history, dim=0)  # [history_length, num_envs, 4]
        if isinstance(ang_vel_history, list):
            ang_vel_history = torch.stack(ang_vel_history, dim=0)  # [history_length, num_envs, 3]
        
        # 确保有足够的历史数据（至少需要delay_steps + history_length）
        total_history_length = quat_history.shape[0]
        if total_history_length < delay_steps + 2:
            # 历史数据不足，返回零
            num_envs = quat_history.shape[1] if quat_history.shape[0] > 0 else 1
            device = next(self.parameters()).device
            return (
                torch.zeros(num_envs, device=device),
                torch.zeros(num_envs, device=device),
                torch.zeros(num_envs, device=device),
                torch.zeros(num_envs, device=device)
            )
        
        # 使用t-delay_steps之前的数据（不包括当前时刻）
        # 例如：如果delay_steps=5，当前是t时刻，我们使用t-5之前的所有数据
        # 取从开始到t-delay_steps的数据
        cutoff_idx = total_history_length - delay_steps
        if cutoff_idx <= 0:
            cutoff_idx = 1  # 至少保留1个数据点
        
        delayed_quat = quat_history[:cutoff_idx]  # [cutoff_idx, num_envs, 4]
        delayed_ang_vel = ang_vel_history[:cutoff_idx]  # [cutoff_idx, num_envs, 3]
        
        # 限制使用的历史长度
        actual_history_length = min(self.history_length, delayed_quat.shape[0])
        if actual_history_length < 2:
            num_envs = delayed_quat.shape[1] if delayed_quat.shape[0] > 0 else 1
            device = next(self.parameters()).device
            return (
                torch.zeros(num_envs, device=device),
                torch.zeros(num_envs, device=device),
                torch.zeros(num_envs, device=device),
                torch.zeros(num_envs, device=device)
            )
        
        # 提取最近的历史数据
        recent_quat = delayed_quat[-actual_history_length:]  # [actual_history_length, num_envs, 4]
        recent_ang_vel = delayed_ang_vel[-actual_history_length:]  # [actual_history_length, num_envs, 3]
        
        num_envs = recent_quat.shape[1]
        
        # 提取roll和pitch角度
        history_roll = torch.zeros(actual_history_length, num_envs, device=recent_quat.device)
        history_pitch = torch.zeros(actual_history_length, num_envs, device=recent_quat.device)
        
        for i in range(actual_history_length):
            roll_i, pitch_i, _ = euler_xyz_from_quat(recent_quat[i])
            history_roll[i] = roll_i
            history_pitch[i] = pitch_i
        
        # 提取roll和pitch角速度
        history_roll_ang_vel = recent_ang_vel[:, :, 0]  # [actual_history_length, num_envs]
        history_pitch_ang_vel = recent_ang_vel[:, :, 1]  # [actual_history_length, num_envs]
        
        # 转置为 [num_envs, actual_history_length]
        history_roll = history_roll.transpose(0, 1)
        history_pitch = history_pitch.transpose(0, 1)
        history_roll_ang_vel = history_roll_ang_vel.transpose(0, 1)
        history_pitch_ang_vel = history_pitch_ang_vel.transpose(0, 1)
        
        # 如果历史长度不足，用第一个值填充
        if actual_history_length < self.history_length:
            padding_size = self.history_length - actual_history_length
            history_roll = torch.cat([
                history_roll[:, 0:1].expand(-1, padding_size),
                history_roll
            ], dim=1)
            history_pitch = torch.cat([
                history_pitch[:, 0:1].expand(-1, padding_size),
                history_pitch
            ], dim=1)
            history_roll_ang_vel = torch.cat([
                history_roll_ang_vel[:, 0:1].expand(-1, padding_size),
                history_roll_ang_vel
            ], dim=1)
            history_pitch_ang_vel = torch.cat([
                history_pitch_ang_vel[:, 0:1].expand(-1, padding_size),
                history_pitch_ang_vel
            ], dim=1)
        
        # 预测当前时刻（基于延迟的历史数据）
        self.eval()
        with torch.no_grad():
            predicted_roll, predicted_pitch, predicted_roll_ang_vel, predicted_pitch_ang_vel = self.forward(
                history_roll, history_pitch, history_roll_ang_vel, history_pitch_ang_vel
            )
        
        return predicted_roll, predicted_pitch, predicted_roll_ang_vel, predicted_pitch_ang_vel
    
    def update(self, platform_history: dict, 
               actual_roll: torch.Tensor, actual_pitch: torch.Tensor,
               actual_roll_ang_vel: torch.Tensor, actual_pitch_ang_vel: torch.Tensor):
        """使用实际平台状态更新网络参数（在线学习）
        
        Args:
            platform_history: 平台历史数据（用于预测，可以是列表或tensor）
            actual_roll: 实际roll角度 [num_envs]
            actual_pitch: 实际pitch角度 [num_envs]
            actual_roll_ang_vel: 实际roll角速度 [num_envs]
            actual_pitch_ang_vel: 实际pitch角速度 [num_envs]
        """
        from isaaclab.utils.math import euler_xyz_from_quat
        
        if platform_history.get('quat_w') is None or platform_history.get('ang_vel_w') is None:
            return
        
        quat_history = platform_history['quat_w']
        ang_vel_history = platform_history['ang_vel_w']
        
        # 处理列表或tensor格式
        if isinstance(quat_history, list):
            quat_history = torch.stack(quat_history, dim=0)
        if isinstance(ang_vel_history, list):
            ang_vel_history = torch.stack(ang_vel_history, dim=0)
        
        # 限制历史长度
        actual_history_length = min(self.history_length, quat_history.shape[0])
        if actual_history_length < 2:
            return
        
        # 提取历史数据（与predict方法相同）
        recent_quat = quat_history[-actual_history_length:]
        recent_ang_vel = ang_vel_history[-actual_history_length:]
        
        num_envs = recent_quat.shape[1]
        
        # 提取roll和pitch角度
        history_roll = torch.zeros(actual_history_length, num_envs, device=recent_quat.device)
        history_pitch = torch.zeros(actual_history_length, num_envs, device=recent_quat.device)
        
        for i in range(actual_history_length):
            roll_i, pitch_i, _ = euler_xyz_from_quat(recent_quat[i])
            history_roll[i] = roll_i
            history_pitch[i] = pitch_i
        
        # 提取roll和pitch角速度
        history_roll_ang_vel = recent_ang_vel[:, :, 0]
        history_pitch_ang_vel = recent_ang_vel[:, :, 1]
        
        # 转置
        history_roll = history_roll.transpose(0, 1)
        history_pitch = history_pitch.transpose(0, 1)
        history_roll_ang_vel = history_roll_ang_vel.transpose(0, 1)
        history_pitch_ang_vel = history_pitch_ang_vel.transpose(0, 1)
        
        # 填充（如果需要）
        if actual_history_length < self.history_length:
            padding_size = self.history_length - actual_history_length
            history_roll = torch.cat([
                history_roll[:, 0:1].expand(-1, padding_size),
                history_roll
            ], dim=1)
            history_pitch = torch.cat([
                history_pitch[:, 0:1].expand(-1, padding_size),
                history_pitch
            ], dim=1)
            history_roll_ang_vel = torch.cat([
                history_roll_ang_vel[:, 0:1].expand(-1, padding_size),
                history_roll_ang_vel
            ], dim=1)
            history_pitch_ang_vel = torch.cat([
                history_pitch_ang_vel[:, 0:1].expand(-1, padding_size),
                history_pitch_ang_vel
            ], dim=1)
        
        # 准备输入和目标（为每个环境分别存储）
        # history_roll等是 [num_envs, history_length]
        # 我们需要为每个环境分别存储数据
        # 但是为了控制缓冲区大小，我们只随机选择部分环境
        num_envs = history_roll.shape[0]
        # 增加每次更新的样本数：从32增加到64，加快训练速度
        # 这样可以更快地收集训练数据，提高预测精度
        max_samples_per_update = min(64, num_envs)
        selected_envs = torch.randperm(num_envs)[:max_samples_per_update]
        
        for env_idx in selected_envs:
            env_inputs = (
                history_roll[env_idx:env_idx+1],  # [1, history_length]
                history_pitch[env_idx:env_idx+1],
                history_roll_ang_vel[env_idx:env_idx+1],
                history_pitch_ang_vel[env_idx:env_idx+1]
            )
            env_targets = torch.stack([
                actual_roll[env_idx:env_idx+1],
                actual_pitch[env_idx:env_idx+1],
                actual_roll_ang_vel[env_idx:env_idx+1],
                actual_pitch_ang_vel[env_idx:env_idx+1]
            ], dim=1)  # [1, 4]
            
            # 添加到训练缓冲区
            self.training_buffer['inputs'].append(env_inputs)
            self.training_buffer['targets'].append(env_targets)
        
        # 限制缓冲区大小
        while len(self.training_buffer['inputs']) > self.training_buffer['max_buffer_size']:
            self.training_buffer['inputs'].pop(0)
            self.training_buffer['targets'].pop(0)
        
        # 提高训练频率：每次有足够样本就训练，不等待
        # 这样可以更快地适应平台运动模式，提高预测精度
        self.train_step_count += 1
        if len(self.training_buffer['inputs']) >= self.batch_size:
            self._train_batch()
    
    def _train_batch(self):
        """批量训练网络"""
        if len(self.training_buffer['inputs']) < self.batch_size:
            return
        
        # 随机选择一批样本
        indices = torch.randint(0, len(self.training_buffer['inputs']), (self.batch_size,), device=self.device)
        
        # 准备批量数据
        # 每个样本的inputs是 (history_roll[1, history_length], history_pitch[1, history_length], ...)
        # 我们需要stack成 [batch_size, history_length] 的形状
        # 注意：输入数据不需要梯度，只有网络参数需要梯度
        batch_history_roll = torch.cat([self.training_buffer['inputs'][i][0] for i in indices], dim=0)  # [batch_size, history_length]
        batch_history_pitch = torch.cat([self.training_buffer['inputs'][i][1] for i in indices], dim=0)
        batch_history_roll_ang_vel = torch.cat([self.training_buffer['inputs'][i][2] for i in indices], dim=0)
        batch_history_pitch_ang_vel = torch.cat([self.training_buffer['inputs'][i][3] for i in indices], dim=0)
        
        # 确保数据在正确的设备上，并detach（输入数据不需要梯度）
        batch_history_roll = batch_history_roll.detach().to(self.device)
        batch_history_pitch = batch_history_pitch.detach().to(self.device)
        batch_history_roll_ang_vel = batch_history_roll_ang_vel.detach().to(self.device)
        batch_history_pitch_ang_vel = batch_history_pitch_ang_vel.detach().to(self.device)
        batch_targets = torch.cat([self.training_buffer['targets'][i] for i in indices], dim=0).detach().to(self.device)  # [batch_size, 4]
        
        batch_inputs = (
            batch_history_roll,
            batch_history_pitch,
            batch_history_roll_ang_vel,
            batch_history_pitch_ang_vel
        )
        
        # 训练模式
        self.train()
        self.optimizer.zero_grad()
        
        # 确保网络参数需要梯度
        for param in self.parameters():
            if not param.requires_grad:
                param.requires_grad = True
        
        # 前向传播
        predicted = self.forward(*batch_inputs)
        predicted_tensor = torch.stack(predicted, dim=1)  # [batch_size, 4]
        
        # 计算损失
        loss = self.criterion(predicted_tensor, batch_targets)
        
        # 检查loss是否有梯度
        if not loss.requires_grad:
            # 如果loss没有梯度，检查网络参数
            has_grad = any(p.requires_grad for p in self.parameters())
            if not has_grad:
                # 如果所有参数都不需要梯度，设置它们需要梯度
                for param in self.parameters():
                    param.requires_grad = True
                # 重新计算loss
                predicted = self.forward(*batch_inputs)
                predicted_tensor = torch.stack(predicted, dim=1)
                loss = self.criterion(predicted_tensor, batch_targets)
            else:
                # 如果参数需要梯度但loss没有，可能是输入问题
                # 这种情况下，我们跳过这次训练
                return
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # 更新参数
        self.optimizer.step()
        
        # 清空缓冲区（可选：保留一些样本用于下次训练）
        # 这里我们保留最后一半的样本
        keep_size = len(self.training_buffer['inputs']) // 2
        if keep_size > 0:
            self.training_buffer['inputs'] = self.training_buffer['inputs'][-keep_size:]
            self.training_buffer['targets'] = self.training_buffer['targets'][-keep_size:]

