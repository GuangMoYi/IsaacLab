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


class ResidualBlock(nn.Module):
    """ResNet风格的残差块，用于提高网络表达能力
    
    使用LayerNorm替代BatchNorm，避免batch_size=1时的问题
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.ln1 = nn.LayerNorm(out_features)  # 使用LayerNorm替代BatchNorm，避免batch_size=1的问题
        self.fc2 = nn.Linear(out_features, out_features)
        self.ln2 = nn.LayerNorm(out_features)  # 使用LayerNorm替代BatchNorm
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
        # 如果输入输出维度不同，需要投影层
        if in_features != out_features:
            self.projection = nn.Linear(in_features, out_features)
        else:
            self.projection = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：x -> fc1 -> ln1 -> relu -> fc2 -> ln2 -> +x -> relu"""
        identity = x
        
        # 如果维度不同，先投影
        if self.projection is not None:
            identity = self.projection(identity)
        
        # 第一个全连接层
        out = self.fc1(x)
        out = self.ln1(out)  # LayerNorm对batch_size不敏感
        out = self.relu(out)
        out = self.dropout(out)
        
        # 第二个全连接层
        out = self.fc2(out)
        out = self.ln2(out)  # LayerNorm对batch_size不敏感
        
        # 残差连接
        out = out + identity
        out = self.relu(out)
        
        return out


class PlatformMotionPredictor(nn.Module):
    """神经网络平台运动预测器（简化版）
    
    输入：机器狗观测历史（所有观测信息）
    - base_lin_vel, base_ang_vel, projected_gravity
    - velocity_commands, joint_pos, joint_vel, actions
    - 时间编码
    
    输出：预测的未来平台运动（roll, pitch, roll_ang_vel, pitch_ang_vel）
    
    支持在线学习：使用实际平台状态更新网络参数
    """
    
    def __init__(
        self,
        history_length: int = 50,
        num_joints: int = 12,
        num_actions: int = 12,
        num_velocity_commands: int = 3,
        num_height_scan_points: int = 0,  # 已废弃，保留仅为兼容性
        hidden_size: int = 256,
        num_layers: int = 3,
        prediction_horizon: float = 0.1,
        prediction_steps: int = 5,  # 预测未来多少步
        learning_rate: float = 5e-3,  # 从1e-3增加到5e-3（提高5倍，帮助突破0.0028的限制）
        device: str = "cuda",
        init_model_versions: bool = True,  # 是否初始化模型版本（避免递归）
    ):
        """初始化预测器
        
        Args:
            history_length: 使用的历史长度（步数）
            num_joints: 关节数量
            num_actions: 动作数量
            num_velocity_commands: 速度命令维度
            num_height_scan_points: 高度扫描点数（已废弃，不再使用）
            hidden_size: 隐藏层大小
            num_layers: LSTM层数
            prediction_horizon: 预测时间范围（秒）
            prediction_steps: 预测未来多少步
            learning_rate: 学习率
            device: 设备（cuda/cpu）
            init_model_versions: 是否初始化模型版本（生产模型和候选模型），默认True
        """
        super().__init__()
        
        self.history_length = history_length
        self.prediction_horizon = prediction_horizon
        self.prediction_steps = prediction_steps
        self.device = device
        self.learning_rate = learning_rate
        
        # 输入维度：机器狗的所有观测信息（不包含height_scan）
        # base_lin_vel: 3
        # base_ang_vel: 3
        # projected_gravity: 3
        # velocity_commands: num_velocity_commands
        # joint_pos: num_joints
        # joint_vel: num_joints
        # actions: num_actions
        # 运动趋势特征（新增）:
        #   base_lin_vel_diff: 3 (线速度一阶差分)
        #   base_ang_vel_diff: 3 (角速度一阶差分)
        #   base_lin_vel_diff2: 3 (线速度二阶差分/加速度)
        #   base_ang_vel_diff2: 3 (角速度二阶差分/角加速度)
        # 时间编码: 6 (sin/cos编码，3个周期)
        input_size = 3 + 3 + 3 + num_velocity_commands + num_joints + num_joints + num_actions + 3 + 3 + 3 + 3 + 6
        # 总共: 9 + num_velocity_commands + 2*num_joints + num_actions + 12(趋势特征) + 6 = 27 + num_velocity_commands + 2*num_joints + num_actions
        
        # LSTM层：处理时序数据
        # 关键改进：保持原有层数，通过其他方法提高性能
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=max(num_layers, 4),  # 保持4层，不增加计算量
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
            bidirectional=True  # 使用双向LSTM，更好地捕捉复杂运动模式
        )
        # 双向LSTM的输出维度是 2 * hidden_size
        lstm_output_size = hidden_size * 2
        
        # 注意力机制：关注历史序列中的重要时刻
        # 关键改进：保持合理的注意力头数，通过其他方法提高性能
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,  # 使用LSTM的输出维度
            num_heads=16,  # 保持16个注意力头，平衡性能和计算量
            dropout=0.1,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(lstm_output_size)
        
        # 输出层：使用ResNet风格的残差连接提高表达能力
        # 输出：未来prediction_steps步的姿态和角速度
        # 每步4个值：[roll, pitch, roll_ang_vel, pitch_ang_vel]
        
        # 第一层：将LSTM输出映射到特征空间
        fc_hidden_size = hidden_size * 4  # 保持4倍特征空间，不增加计算量
        self.fc1 = nn.Linear(lstm_output_size, fc_hidden_size)
        self.ln1 = nn.LayerNorm(fc_hidden_size)  # 使用LayerNorm替代BatchNorm，避免batch_size=1的问题
        
        # ResNet风格的残差块（保持原有数量，通过其他方法提高性能）
        self.res_block1 = ResidualBlock(fc_hidden_size, fc_hidden_size)
        self.res_block2 = ResidualBlock(fc_hidden_size, fc_hidden_size)
        self.res_block3 = ResidualBlock(fc_hidden_size, fc_hidden_size)
        self.res_block4 = ResidualBlock(fc_hidden_size, fc_hidden_size)  # 保持4个残差块
        
        # 最终输出层
        # 输出2维：roll, pitch（只预测这两个角度，因为跟随任务只需要保持XY平面平行）
        # 注意：x, y, z, yaw对跟随任务没有帮助，反而可能引入噪声，增加学习难度
        self.fc_out = nn.Linear(fc_hidden_size, prediction_steps * 2)
        
        # Dropout层（适当增加dropout率，防止过拟合）
        self.dropout = nn.Dropout(0.15)  # 从0.1增加到0.15，防止过拟合
        
        # 优化器：使用AdamW（权重衰减）和更稳定的学习率调度
        self.optimizer = optim.AdamW(
            self.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-5,
            betas=(0.9, 0.999),  # 默认beta值
            eps=1e-8
        )
        
        # 学习率调度器：使用Warmup + 余弦退火重启，帮助跳出局部最优
        # Warmup阶段：前100步线性增加学习率，帮助模型稳定训练
        # 然后使用余弦退火重启，每500步重启一次，帮助跳出局部最优
        # 关键改进：降低重启周期，更频繁地重启，帮助跳出局部最优
        self.warmup_steps = 100
        self.restart_interval = 500  # 每500步重启一次学习率（在_train_batch中会重新设置）
        self.scheduler = None  # 延迟初始化，在第一次训练时创建
        self._scheduler_initialized = False
        
        # 数据归一化参数（用于输入和输出的归一化，提高训练稳定性）
        # 关键改进：使用自适应归一化，根据实际数据范围动态调整
        # 初始值：使用较大的范围，避免数据超出范围
        # 2维归一化参数：roll, pitch（只预测这两个角度）
        self.input_scale = {
            'roll': 1.0 / 1.0,    # 姿态归一化范围：±1.0 rad
            'pitch': 1.0 / 1.0,
            'roll_trend': 1.0 / 0.01,  # 趋势归一化
            'pitch_trend': 1.0 / 0.01,
            'time': 1.0,  # 时间归一化：时间特征已经在[0, 1]范围内，不需要额外缩放
        }
        self.output_scale = {
            'roll': 1.0,
            'pitch': 1.0,
        }
        
        # 自适应归一化：跟踪实际数据的最大值，动态调整归一化参数
        self.data_stats = {
            'roll': {'max_abs': 0.0, 'count': 0},
            'pitch': {'max_abs': 0.0, 'count': 0},
        }
        self.adaptive_normalization_enabled = True
        self.stats_update_frequency = 100  # 每100步更新一次统计信息
        
        # 关键修复：保存归一化统计信息（用于预测时的归一化和反归一化）
        # 观测数据的归一化统计（动态更新）
        self.obs_stats = {
            'base_lin_vel': {'mean': None, 'std': None, 'count': 0},
            'base_ang_vel': {'mean': None, 'std': None, 'count': 0},
            'projected_gravity': {'mean': None, 'std': None, 'count': 0},
            'velocity_commands': {'mean': None, 'std': None, 'count': 0},
            'joint_pos': {'mean': None, 'std': None, 'count': 0},
            'joint_vel': {'mean': None, 'std': None, 'count': 0},
            'actions': {'mean': None, 'std': None, 'count': 0},
        }
        # 目标数据的归一化统计（动态更新）
        self.target_stats = {'mean': None, 'std': None, 'count': 0}
        
        # 损失函数：使用Huber Loss（对异常值更鲁棒）+ MSE Loss的组合
        # Huber Loss在误差较小时表现更好，MSE Loss在误差较大时提供更强的梯度
        self.criterion_mse = nn.MSELoss(reduction='none')
        self.criterion_huber = nn.SmoothL1Loss(reduction='none')
        self.huber_delta = 0.1  # Huber Loss的阈值
        
        # 训练历史（用于批量更新，支持长期训练）
        self.training_buffer = {
            'inputs': [],
            'targets': [],
            'max_buffer_size': 20000,  # 大幅增加缓冲区大小：从10000增加到20000，支持更多训练样本
        }
        
        # 训练计数器
        self.train_step_count = 0
        self.batch_size = 128  # 增加批量大小：从64增加到128，提高训练稳定性
        self.max_samples_per_update = 128  # 增加每次更新的样本数
        
        # 困难样本挖掘：保留预测误差较大的样本，帮助模型学习困难案例
        self.hard_sample_ratio = 0.3  # 30%的样本来自困难样本
        self.hard_sample_buffer = {
            'inputs': [],
            'targets': [],
            'errors': [],  # 存储每个样本的预测误差
            'max_size': 200,  # 困难样本缓冲区大小
        }
        
        # 数据增强：添加小量噪声提高模型鲁棒性
        self.use_data_augmentation = True
        self.noise_std = 0.01  # 噪声标准差（归一化后的值）
        
        # ========== 模型版本控制机制 ==========
        # 维护两个模型：
        # 1. candidate_model: 候选模型，用于训练和更新（直接使用self，不创建引用避免循环）
        # 2. production_model: 生产模型，用于实际预测和奖励计算
        # 只有候选模型评估通过后，才会替换生产模型
        # 注意：不设置 self.candidate_model = self，避免循环引用导致递归错误
        self.production_model = None  # 生产模型（用于预测，延迟初始化）
        self._production_model_initialized = False  # 标记生产模型是否已初始化
        
        # 评估相关参数
        self.evaluation_interval = 1000  # 每隔多少步评估一次候选模型（从200增加到1000，降低5倍，减少评估开销）
        self.last_evaluation_step = 0  # 上次评估的步数
        self.candidate_train_steps = 0  # 候选模型训练步数
        
        # 损失历史记录（用于学习率调整和训练监控）
        self.loss_history = []
        self.loss_history_window = 500  # 记录最近500步的损失
        
        # 延迟初始化生产模型（在to(device)之后）
        # 不在__init__中初始化，避免递归问题
        if init_model_versions:
            # 标记需要初始化，但延迟到to(device)之后
            self._need_init_production_model = True
        else:
            self._need_init_production_model = False
        
        # ========== 预测质量评估机制 ==========
        # 使用历史数据验证网络预测能力，而不是每次训练都检查
        # 一旦评估通过，就持续使用网络预测（不再每次都检查）
        self.prediction_quality_verified = False  # 预测质量是否已验证通过
        # 关键修复：调整阈值，使其更合理
        # 对于复杂船舶运动，0.01 rad（约0.57度）仍然太严格
        # 从日志看，平均误差在0.23-0.43 rad之间，所以阈值应该设置为0.05 rad（约2.87度）
        # 这是一个更合理的预测误差阈值，既能保证预测质量，又不会太严格
        # 注意：这个阈值是用于评估预测器是否"足够好"的，不是最终目标
        self.prediction_quality_threshold = 0.05  # 预测质量阈值（rad）：平均误差小于此值认为预测质量足够好（从0.01增加到0.05，更合理）
        self.min_evaluation_samples = 50  # 评估时至少需要50个样本点
        self.evaluation_accuracy_ratio = 0.80  # 至少80%的预测误差小于阈值才认为网络训练好了（从95%降低到80%，更合理）
        # ======================================
    
    def _init_model_versions(self):
        """初始化候选模型和生产模型
        
        两个模型初始参数相同，但后续会独立更新：
        - candidate_model: 用于训练，持续更新参数（直接使用self，不创建引用）
        - production_model: 用于预测，只有候选模型评估通过后才更新
        """
        # 如果生产模型已存在，先删除它（避免结构不匹配的问题）
        if self.production_model is not None:
            del self.production_model
            self.production_model = None
        
        # 候选模型：用于训练和更新
        # 注意：不设置 self.candidate_model = self，避免循环引用
        # 候选模型就是 self 本身，在训练时直接使用 self
        
        # 生产模型：用于实际预测
        # 手动创建模型结构（避免创建完整的PlatformMotionPredictor实例导致递归）
        # 只创建必要的网络层，不创建优化器等
        # 优化：与主模型结构完全一致（使用机器狗观测历史作为输入）
        # 从主模型的LSTM获取实际的input_size（因为输入维度已经改变）
        input_size = self.lstm.input_size  # 从主模型获取实际输入维度
        hidden_size = self.lstm.hidden_size
        num_layers = self.lstm.num_layers
        is_bidirectional = self.lstm.bidirectional
        lstm_output_size = hidden_size * 2 if is_bidirectional else hidden_size  # 双向LSTM输出维度是2倍
        fc_hidden_size = lstm_output_size  # 与LSTM输出维度一致
        
        # 创建生产模型的LSTM层（与主模型完全一致：双向LSTM）
        production_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.15 if num_layers > 1 else 0.0,  # 与主模型一致
            bidirectional=is_bidirectional  # 与主模型一致，使用双向LSTM
        )
        
        # 创建生产模型的注意力机制（与主模型一致）
        # 关键修复：从主模型获取实际的num_heads，确保参数大小匹配
        num_heads = self.attention.num_heads  # 从主模型获取实际的注意力头数
        production_attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,  # 使用LSTM的输出维度
            num_heads=num_heads,  # 与主模型一致，从主模型获取
            dropout=0.1,
            batch_first=True
        )
        production_attention_norm = nn.LayerNorm(lstm_output_size)
        
        # 创建生产模型的全连接层（与主模型结构完全一致，使用ResNet风格）
        # 关键修复：从主模型获取实际的fc_hidden_size，确保参数大小匹配
        fc_hidden_size = self.fc1.out_features  # 从主模型获取实际的fc_hidden_size
        production_fc1 = nn.Linear(lstm_output_size, fc_hidden_size)
        production_ln1 = nn.LayerNorm(fc_hidden_size)  # 使用LayerNorm替代BatchNorm
        production_res_block1 = ResidualBlock(fc_hidden_size, fc_hidden_size)
        production_res_block2 = ResidualBlock(fc_hidden_size, fc_hidden_size)
        production_res_block3 = ResidualBlock(fc_hidden_size, fc_hidden_size)
        # 关键修复：检查主模型是否有res_block4，如果有则创建
        production_res_block4 = None
        if hasattr(self, 'res_block4'):
            production_res_block4 = ResidualBlock(fc_hidden_size, fc_hidden_size)
        production_fc_out = nn.Linear(fc_hidden_size, self.prediction_steps * 2)
        production_dropout = nn.Dropout(0.15)  # 与主模型一致（从0.1改为0.15）
        
        # 创建一个简单的容器来存储生产模型的层
        # 使用nn.ModuleDict来存储，这样PyTorch可以正确管理
        # 注意：先创建ModuleDict，然后添加子模块，避免在创建时出现问题
        self.production_model = nn.ModuleDict()
        self.production_model['lstm'] = production_lstm
        self.production_model['attention'] = production_attention
        self.production_model['attention_norm'] = production_attention_norm
        self.production_model['fc1'] = production_fc1
        self.production_model['ln1'] = production_ln1  # LayerNorm替代BatchNorm
        self.production_model['res_block1'] = production_res_block1
        self.production_model['res_block2'] = production_res_block2
        self.production_model['res_block3'] = production_res_block3
        if production_res_block4 is not None:
            self.production_model['res_block4'] = production_res_block4
        self.production_model['fc_out'] = production_fc_out
        self.production_model['dropout'] = production_dropout
        
        # 注意：不在初始化时调用to(device)，避免递归
        # 生产模型会在主模型的to(device)调用时自动移动到设备
        
        # 复制参数（手动复制，避免递归问题）
        # 关键修复：检查参数形状是否匹配，如果不匹配则跳过（可能是初始化时的临时状态）
        with torch.no_grad():
            # 复制LSTM参数
            for prod_param, cand_param in zip(self.production_model['lstm'].parameters(), self.lstm.parameters()):
                if prod_param.shape == cand_param.shape:
                    prod_param.data.copy_(cand_param.data)
                else:
                    # 如果形状不匹配，使用随机初始化（这种情况不应该发生，但作为安全措施）
                    print(f"Warning: LSTM parameter shape mismatch: prod={prod_param.shape}, cand={cand_param.shape}")
            # 复制FC层参数（ResNet风格）
            for prod_param, cand_param in zip(self.production_model['fc1'].parameters(), self.fc1.parameters()):
                if prod_param.shape == cand_param.shape:
                    prod_param.data.copy_(cand_param.data)
                else:
                    print(f"Warning: FC1 parameter shape mismatch: prod={prod_param.shape}, cand={cand_param.shape}")
            for prod_param, cand_param in zip(self.production_model['ln1'].parameters(), self.ln1.parameters()):
                if prod_param.shape == cand_param.shape:
                    prod_param.data.copy_(cand_param.data)
                else:
                    print(f"Warning: LN1 parameter shape mismatch: prod={prod_param.shape}, cand={cand_param.shape}")
            for prod_param, cand_param in zip(self.production_model['attention'].parameters(), self.attention.parameters()):
                if prod_param.shape == cand_param.shape:
                    prod_param.data.copy_(cand_param.data)
                else:
                    print(f"Warning: Attention parameter shape mismatch: prod={prod_param.shape}, cand={cand_param.shape}")
            for prod_param, cand_param in zip(self.production_model['attention_norm'].parameters(), self.attention_norm.parameters()):
                if prod_param.shape == cand_param.shape:
                    prod_param.data.copy_(cand_param.data)
                else:
                    print(f"Warning: AttentionNorm parameter shape mismatch: prod={prod_param.shape}, cand={cand_param.shape}")
            for prod_param, cand_param in zip(self.production_model['res_block1'].parameters(), self.res_block1.parameters()):
                if prod_param.shape == cand_param.shape:
                    prod_param.data.copy_(cand_param.data)
                else:
                    print(f"Warning: ResBlock1 parameter shape mismatch: prod={prod_param.shape}, cand={cand_param.shape}")
            for prod_param, cand_param in zip(self.production_model['res_block2'].parameters(), self.res_block2.parameters()):
                if prod_param.shape == cand_param.shape:
                    prod_param.data.copy_(cand_param.data)
                else:
                    print(f"Warning: ResBlock2 parameter shape mismatch: prod={prod_param.shape}, cand={cand_param.shape}")
            for prod_param, cand_param in zip(self.production_model['res_block3'].parameters(), self.res_block3.parameters()):
                if prod_param.shape == cand_param.shape:
                    prod_param.data.copy_(cand_param.data)
                else:
                    print(f"Warning: ResBlock3 parameter shape mismatch: prod={prod_param.shape}, cand={cand_param.shape}")
            # 关键修复：如果主模型有res_block4，则复制其参数
            if 'res_block4' in self.production_model and hasattr(self, 'res_block4'):
                for prod_param, cand_param in zip(self.production_model['res_block4'].parameters(), self.res_block4.parameters()):
                    if prod_param.shape == cand_param.shape:
                        prod_param.data.copy_(cand_param.data)
                    else:
                        print(f"Warning: ResBlock4 parameter shape mismatch: prod={prod_param.shape}, cand={cand_param.shape}")
            for prod_param, cand_param in zip(self.production_model['fc_out'].parameters(), self.fc_out.parameters()):
                if prod_param.shape == cand_param.shape:
                    prod_param.data.copy_(cand_param.data)
                else:
                    print(f"Warning: FCOut parameter shape mismatch: prod={prod_param.shape}, cand={cand_param.shape}")
        
        # 生产模型不需要梯度（只用于推理）
        for param in self.production_model.parameters():
            param.requires_grad = False
        self.production_model.eval()
        
        # 标记当前使用的是哪个模型
        self._using_production_model = True  # 初始时使用生产模型（因为两个模型参数相同）
    
    def forward(
        self,
        base_lin_vel: torch.Tensor,  # [batch_size, history_length, 3]
        base_ang_vel: torch.Tensor,  # [batch_size, history_length, 3]
        projected_gravity: torch.Tensor,  # [batch_size, history_length, 3]
        velocity_commands: torch.Tensor,  # [batch_size, history_length, num_velocity_commands]
        joint_pos: torch.Tensor,  # [batch_size, history_length, num_joints]
        joint_vel: torch.Tensor,  # [batch_size, history_length, num_joints]
        actions: torch.Tensor,  # [batch_size, history_length, num_actions]
        history_time: torch.Tensor = None,  # [batch_size, history_length, 6] 或 None
    ) -> torch.Tensor:
        """前向传播：从机器狗观测历史预测未来平台运动
        
        Args:
            base_lin_vel: 基座线速度历史 [batch_size, history_length, 3]
            base_ang_vel: 基座角速度历史 [batch_size, history_length, 3]
            projected_gravity: 投影重力历史 [batch_size, history_length, 3]
            velocity_commands: 速度命令历史 [batch_size, history_length, num_velocity_commands]
            joint_pos: 关节位置历史 [batch_size, history_length, num_joints]
            joint_vel: 关节速度历史 [batch_size, history_length, num_joints]
            actions: 动作历史 [batch_size, history_length, num_actions]
            history_time: 时间编码 [batch_size, history_length, 6] (可选)
        
        Returns:
            predicted_states: 预测的未来多步状态 [batch_size, prediction_steps, 2] (roll, pitch)
                每步包含：[roll, pitch, roll_ang_vel, pitch_ang_vel]
        """
        batch_size = base_lin_vel.shape[0]
        history_length = base_lin_vel.shape[1]
        
        # 添加时间特征（使用sin/cos编码帮助学习周期性）
        if history_time is None:
            device = base_lin_vel.device
            time_indices = torch.arange(history_length, device=device, dtype=base_lin_vel.dtype).unsqueeze(0).expand(batch_size, -1)
            normalized_time = time_indices / max(history_length - 1, 1.0)  # 归一化到[0, 1]
            
            # 使用sin/cos编码，帮助模型学习周期性
            period_ratios = [1.0, 2.0, 4.0]  # 不同周期的倍数
            time_features_list = []
            for period_ratio in period_ratios:
                phase = 2.0 * torch.pi * normalized_time * period_ratio
                time_features_list.append(torch.sin(phase))  # [batch_size, history_length]
                time_features_list.append(torch.cos(phase))  # [batch_size, history_length]
            history_time = torch.stack(time_features_list, dim=-1)  # [batch_size, history_length, 6]
        
        # 关键改进：添加运动趋势特征（一阶和二阶差分），帮助模型学习运动动态
        # 一阶差分（速度）：反映运动趋势
        base_lin_vel_diff = torch.diff(base_lin_vel, dim=1, prepend=base_lin_vel[:, 0:1, :])  # [batch_size, history_length, 3]
        base_ang_vel_diff = torch.diff(base_ang_vel, dim=1, prepend=base_ang_vel[:, 0:1, :])  # [batch_size, history_length, 3]
        
        # 二阶差分（加速度）：反映运动变化率
        base_lin_vel_diff2 = torch.diff(base_lin_vel_diff, dim=1, prepend=base_lin_vel_diff[:, 0:1, :])  # [batch_size, history_length, 3]
        base_ang_vel_diff2 = torch.diff(base_ang_vel_diff, dim=1, prepend=base_ang_vel_diff[:, 0:1, :])  # [batch_size, history_length, 3]
        
        # 组合输入（包含所有机器狗观测信息 + 运动趋势特征）
        inputs = torch.cat([
            base_lin_vel,  # [batch_size, history_length, 3]
            base_ang_vel,  # [batch_size, history_length, 3]
            projected_gravity,  # [batch_size, history_length, 3]
            velocity_commands,  # [batch_size, history_length, num_velocity_commands]
            joint_pos,  # [batch_size, history_length, num_joints]
            joint_vel,  # [batch_size, history_length, num_joints]
            actions,  # [batch_size, history_length, num_actions]
            base_lin_vel_diff,  # [batch_size, history_length, 3] - 新增：线速度趋势
            base_ang_vel_diff,  # [batch_size, history_length, 3] - 新增：角速度趋势
            base_lin_vel_diff2,  # [batch_size, history_length, 3] - 新增：线加速度
            base_ang_vel_diff2,  # [batch_size, history_length, 3] - 新增：角加速度
            history_time,  # [batch_size, history_length, 6]
        ], dim=-1)  # [batch_size, history_length, input_size + 12] (增加了12维趋势特征)
        
        # 确保inputs在正确的设备上
        # 关键：使用.clone()确保不破坏计算图，但输入数据本身不需要梯度，所以这里不需要梯度
        target_device = next(self.parameters()).device
        if inputs.device != target_device:
            inputs = inputs.to(target_device)
        
        # LSTM处理（双向LSTM，输出维度是hidden_size*2）
        lstm_out, _ = self.lstm(inputs)  # [batch_size, history_length, hidden_size*2]
        
        # 注意力机制：关注历史序列中的重要时刻
        # 使用最后一个时间步作为query，所有时间步作为key和value
        last_hidden_expanded = lstm_out[:, -1:, :]  # [batch_size, 1, hidden_size*2]
        attn_out, _ = self.attention(
            last_hidden_expanded,  # query: 最后一个时间步
            lstm_out,  # key: 所有时间步
            lstm_out  # value: 所有时间步
        )  # [batch_size, 1, hidden_size]
        
        # 残差连接和层归一化
        attn_out = self.attention_norm(attn_out.squeeze(1) + lstm_out[:, -1, :])  # [batch_size, hidden_size]
        
        # ResNet风格的输出层（优化：只使用2个残差块）
        # 第一层
        x = self.fc1(attn_out)  # [batch_size, fc_hidden_size]
        x = self.ln1(x)  # LayerNorm替代BatchNorm，避免batch_size=1的问题
        x = torch.relu(x)
        x = self.dropout(x)
        
        # 残差块（保持4个残差块，通过其他方法提高性能）
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)  # 第四个残差块
        
        # 最终输出层
        output = self.fc_out(x)  # [batch_size, prediction_steps * 2]
        
        # 重塑为 [batch_size, prediction_steps, 2]
        # 每步2个值：[roll, pitch]
        predicted_states = output.view(batch_size, self.prediction_steps, 2)
        
        return predicted_states
    
    def to(self, *args, **kwargs):
        """重写to方法，确保生产模型也被正确移动"""
        # 先调用父类的to方法
        result = super().to(*args, **kwargs)
        
        # 如果生产模型还未初始化，现在初始化它（在移动到设备之后）
        if self._need_init_production_model and not self._production_model_initialized:
            self._init_model_versions()
            self._production_model_initialized = True
            self._need_init_production_model = False
        
        # 如果生产模型已初始化，确保它也在正确的设备上
        if self.production_model is not None:
            self.production_model = self.production_model.to(*args, **kwargs)
        
        return result
    
    def _forward_with_model(
        self, 
        model,
        base_lin_vel: torch.Tensor,  # [batch_size, history_length, 3]
        base_ang_vel: torch.Tensor,  # [batch_size, history_length, 3]
        projected_gravity: torch.Tensor,  # [batch_size, history_length, 3]
        velocity_commands: torch.Tensor,  # [batch_size, history_length, num_velocity_commands]
        joint_pos: torch.Tensor,  # [batch_size, history_length, num_joints]
        joint_vel: torch.Tensor,  # [batch_size, history_length, num_joints]
        actions: torch.Tensor,  # [batch_size, history_length, num_actions]
        history_time: torch.Tensor = None,  # [batch_size, history_length, 6] 或 None
    ):
        """使用指定模型进行前向传播（使用机器狗观测历史）
        
        Args:
            model: 要使用的模型（候选模型self或生产模型self.production_model）
            base_lin_vel, base_ang_vel, projected_gravity: 机器狗基座状态
            velocity_commands: 速度命令
            joint_pos, joint_vel: 关节状态
            actions: 动作
            history_time: 时间编码（可选）
        
        Returns:
            predicted_states: 预测的未来多步状态 [batch_size, prediction_steps, 2] (roll, pitch)
        """
        batch_size = base_lin_vel.shape[0]
        history_length = base_lin_vel.shape[1]
        
        # 添加时间特征（使用sin/cos编码帮助学习周期性）
        if history_time is None:
            device = base_lin_vel.device
            time_indices = torch.arange(history_length, device=device, dtype=base_lin_vel.dtype).unsqueeze(0).expand(batch_size, -1)
            normalized_time = time_indices / max(history_length - 1, 1.0)  # 归一化到[0, 1]
            
            # 使用sin/cos编码，帮助模型学习周期性
            period_ratios = [1.0, 2.0, 4.0]  # 不同周期的倍数
            time_features_list = []
            for period_ratio in period_ratios:
                phase = 2.0 * torch.pi * normalized_time * period_ratio
                time_features_list.append(torch.sin(phase))  # [batch_size, history_length]
                time_features_list.append(torch.cos(phase))  # [batch_size, history_length]
            history_time = torch.stack(time_features_list, dim=-1)  # [batch_size, history_length, 6]
        
        # 关键改进：添加运动趋势特征（一阶和二阶差分），帮助模型学习运动动态
        # 一阶差分（速度）：反映运动趋势
        base_lin_vel_diff = torch.diff(base_lin_vel, dim=1, prepend=base_lin_vel[:, 0:1, :])  # [batch_size, history_length, 3]
        base_ang_vel_diff = torch.diff(base_ang_vel, dim=1, prepend=base_ang_vel[:, 0:1, :])  # [batch_size, history_length, 3]
        
        # 二阶差分（加速度）：反映运动变化率
        base_lin_vel_diff2 = torch.diff(base_lin_vel_diff, dim=1, prepend=base_lin_vel_diff[:, 0:1, :])  # [batch_size, history_length, 3]
        base_ang_vel_diff2 = torch.diff(base_ang_vel_diff, dim=1, prepend=base_ang_vel_diff[:, 0:1, :])  # [batch_size, history_length, 3]
        
        # 组合输入（包含所有机器狗观测信息 + 运动趋势特征）
        inputs = torch.cat([
            base_lin_vel,  # [batch_size, history_length, 3]
            base_ang_vel,  # [batch_size, history_length, 3]
            projected_gravity,  # [batch_size, history_length, 3]
            velocity_commands,  # [batch_size, history_length, num_velocity_commands]
            joint_pos,  # [batch_size, history_length, num_joints]
            joint_vel,  # [batch_size, history_length, num_joints]
            actions,  # [batch_size, history_length, num_actions]
            base_lin_vel_diff,  # [batch_size, history_length, 3] - 新增：线速度趋势
            base_ang_vel_diff,  # [batch_size, history_length, 3] - 新增：角速度趋势
            base_lin_vel_diff2,  # [batch_size, history_length, 3] - 新增：线加速度
            base_ang_vel_diff2,  # [batch_size, history_length, 3] - 新增：角加速度
            history_time,  # [batch_size, history_length, 6]
        ], dim=-1)  # [batch_size, history_length, input_size + 12] (增加了12维趋势特征)
        
        inputs = inputs.to(next(model.parameters()).device)
        
        # 使用指定模型进行前向传播
        # 处理两种情况：self（候选模型）或self.production_model（生产模型，是ModuleDict）
        if isinstance(model, nn.ModuleDict):
            # 生产模型：使用ModuleDict（双向LSTM + 注意力 + ResNet风格）
            lstm_out, _ = model['lstm'](inputs)  # [batch_size, history_length, hidden_size*2] (双向)
            last_hidden = lstm_out[:, -1, :]
            
            # 注意力机制
            last_hidden_expanded = lstm_out[:, -1:, :]
            attn_out, _ = model['attention'](last_hidden_expanded, lstm_out, lstm_out)
            attn_out = model['attention_norm'](attn_out.squeeze(1) + last_hidden)
            
            # ResNet风格的前向传播（4个残差块）
            x = model['fc1'](attn_out)
            x = model['ln1'](x)  # LayerNorm替代BatchNorm
            x = torch.relu(x)
            x = model['dropout'](x)
            x = model['res_block1'](x)
            x = model['res_block2'](x)
            x = model['res_block3'](x)
            x = model['res_block4'](x)  # 第四个残差块
            output = model['fc_out'](x)
            
            predicted_states = output.view(batch_size, self.prediction_steps, 2)
        else:
            # 候选模型：直接使用self的属性（双向LSTM + 注意力 + ResNet风格）
            lstm_out, _ = model.lstm(inputs)  # [batch_size, history_length, hidden_size*2] (双向)
            last_hidden = lstm_out[:, -1, :]
            
            # 注意力机制
            last_hidden_expanded = lstm_out[:, -1:, :]
            attn_out, _ = model.attention(last_hidden_expanded, lstm_out, lstm_out)
            attn_out = model.attention_norm(attn_out.squeeze(1) + last_hidden)
            
            # ResNet风格的前向传播（4个残差块）
            x = model.fc1(attn_out)
            x = model.ln1(x)  # LayerNorm替代BatchNorm
            x = torch.relu(x)
            x = model.dropout(x)
            x = model.res_block1(x)
            x = model.res_block2(x)
            x = model.res_block3(x)
            x = model.res_block4(x)  # 第四个残差块
            output = model.fc_out(x)
            
            predicted_states = output.view(batch_size, model.prediction_steps, 2)
        
        return predicted_states
    
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
        
        # 获取环境数量（用于广播）
        num_envs = quat_history.shape[1] if quat_history.shape[0] > 0 else 1
        
        # 关键改进：各环境的运动是完全一致的，所以预测结果只需要预测一个平台的运动
        # 但训练时使用所有环境的数据（增加训练数据量）
        # 预测时只使用环境0的数据，然后广播到所有环境
        recent_quat = quat_history[-actual_history_length:, 0:1, :]  # [actual_history_length, 1, 4] 只使用环境0
        recent_ang_vel = ang_vel_history[-actual_history_length:, 0:1, :]  # [actual_history_length, 1, 3] 只使用环境0
        
        # 提取roll和pitch角度（只提取环境0）
        history_roll = torch.zeros(actual_history_length, 1, device=recent_quat.device)
        history_pitch = torch.zeros(actual_history_length, 1, device=recent_quat.device)
        
        for i in range(actual_history_length):
            roll_i, pitch_i, _ = euler_xyz_from_quat(recent_quat[i])
            history_roll[i, 0] = roll_i[0]
            history_pitch[i, 0] = pitch_i[0]
        
        # 提取roll和pitch角速度（只提取环境0）
        history_roll_ang_vel = recent_ang_vel[:, 0, 0:1]  # [actual_history_length, 1]
        history_pitch_ang_vel = recent_ang_vel[:, 0, 1:2]  # [actual_history_length, 1]
        
        # 转置为 [1, actual_history_length]
        history_roll = history_roll.transpose(0, 1)  # [1, actual_history_length]
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
        
        # 数据归一化（输入）：与训练保持一致
        history_roll_norm = history_roll * self.input_scale['roll']
        history_pitch_norm = history_pitch * self.input_scale['pitch']
        history_roll_ang_vel_norm = history_roll_ang_vel * self.input_scale['roll_ang_vel']
        history_pitch_ang_vel_norm = history_pitch_ang_vel * self.input_scale['pitch_ang_vel']
        
        # 预测未来多步（使用生产模型，只预测环境0，然后广播到所有环境）
        # 如果生产模型未初始化，使用候选模型（self）
        if self.production_model is None:
            self.eval()
            with torch.no_grad():
                predicted_states_norm = self.forward(
                    history_roll_norm, history_pitch_norm, history_roll_ang_vel_norm, history_pitch_ang_vel_norm
                )  # [1, prediction_steps, 2]
        else:
            self.production_model.eval()
            with torch.no_grad():
                predicted_states_norm = self._forward_with_model(
                    self.production_model,
                    history_roll_norm, history_pitch_norm, history_roll_ang_vel_norm, history_pitch_ang_vel_norm
                )  # [1, prediction_steps, 2]
        
        # 反归一化输出（只包含roll和pitch，2维）
        predicted_states = predicted_states_norm.clone()
        predicted_states[:, :, 0] = predicted_states_norm[:, :, 0] / self.input_scale['roll']
        predicted_states[:, :, 1] = predicted_states_norm[:, :, 1] / self.input_scale['pitch']
        # 注意：不再包含角速度反归一化，因为输出只包含roll和pitch（2维）
        
        # 选择第1步的预测（下一步）
        predicted_state = predicted_states[0, 0, :]  # [2] (roll, pitch)
        
        # 广播到所有环境（因为各环境的运动是完全一致的）
        predicted_roll = predicted_state[0].expand(num_envs)
        predicted_pitch = predicted_state[1].expand(num_envs)
        # 角速度暂时设为0（如果需要可以后续添加）
        predicted_roll_ang_vel = torch.zeros_like(predicted_roll)
        predicted_pitch_ang_vel = torch.zeros_like(predicted_roll)
        
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
        
        # 获取环境数量（用于广播）
        num_envs = quat_history.shape[1] if quat_history.shape[0] > 0 else 1
        
        # 关键改进：各环境的运动是完全一致的，所以预测结果只需要预测一个平台的运动
        # 但训练时使用所有环境的数据（增加训练数据量）
        # 预测时只使用环境0的数据，然后广播到所有环境
        recent_quat = delayed_quat[-actual_history_length:, 0:1, :]  # [actual_history_length, 1, 4] 只使用环境0
        recent_ang_vel = delayed_ang_vel[-actual_history_length:, 0:1, :]  # [actual_history_length, 1, 3] 只使用环境0
        
        # 提取roll和pitch角度（只提取环境0）
        history_roll = torch.zeros(actual_history_length, 1, device=recent_quat.device)
        history_pitch = torch.zeros(actual_history_length, 1, device=recent_quat.device)
        
        for i in range(actual_history_length):
            roll_i, pitch_i, _ = euler_xyz_from_quat(recent_quat[i])
            history_roll[i, 0] = roll_i[0]
            history_pitch[i, 0] = pitch_i[0]
        
        # 提取roll和pitch角速度（只提取环境0）
        history_roll_ang_vel = recent_ang_vel[:, 0, 0:1]  # [actual_history_length, 1]
        history_pitch_ang_vel = recent_ang_vel[:, 0, 1:2]  # [actual_history_length, 1]
        
        # 转置为 [1, actual_history_length]
        history_roll = history_roll.transpose(0, 1)  # [1, actual_history_length]
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
        
        # 数据归一化（输入）：与训练保持一致
        history_roll_norm = history_roll * self.input_scale['roll']
        history_pitch_norm = history_pitch * self.input_scale['pitch']
        history_roll_ang_vel_norm = history_roll_ang_vel * self.input_scale['roll_ang_vel']
        history_pitch_ang_vel_norm = history_pitch_ang_vel * self.input_scale['pitch_ang_vel']
        
        # 预测未来多步（使用生产模型，基于延迟的历史数据，只预测环境0，然后广播到所有环境）
        # 如果生产模型未初始化，使用候选模型（self）
        if self.production_model is None:
            self.eval()
            with torch.no_grad():
                predicted_states_norm = self.forward(
                    history_roll_norm, history_pitch_norm, history_roll_ang_vel_norm, history_pitch_ang_vel_norm
                )  # [1, prediction_steps, 2]
        else:
            self.production_model.eval()
            with torch.no_grad():
                predicted_states_norm = self._forward_with_model(
                    self.production_model,
                    history_roll_norm, history_pitch_norm, history_roll_ang_vel_norm, history_pitch_ang_vel_norm
                )  # [1, prediction_steps, 2]
        
        # 反归一化输出（只包含roll和pitch，2维）
        predicted_states = predicted_states_norm.clone()
        predicted_states[:, :, 0] = predicted_states_norm[:, :, 0] / self.input_scale['roll']
        predicted_states[:, :, 1] = predicted_states_norm[:, :, 1] / self.input_scale['pitch']
        # 注意：不再包含角速度反归一化，因为输出只包含roll和pitch（2维）
        
        # 选择第delay_steps步的预测（对应当前时刻）
        # 如果delay_steps > prediction_steps，选择最后一步
        step_idx = min(delay_steps, self.prediction_steps - 1)
        
        # 提取对应步数的预测（环境0）
        predicted_state = predicted_states[0, step_idx, :]  # [2] (roll, pitch)
        
        # 广播到所有环境（因为各环境的运动是完全一致的）
        predicted_roll = predicted_state[0].expand(num_envs)
        predicted_pitch = predicted_state[1].expand(num_envs)
        # 角速度暂时设为0（如果需要可以后续添加）
        predicted_roll_ang_vel = torch.zeros_like(predicted_roll)
        predicted_pitch_ang_vel = torch.zeros_like(predicted_roll)
        
        return predicted_roll, predicted_pitch, predicted_roll_ang_vel, predicted_pitch_ang_vel
    
    def predict_future(
        self,
        platform_history: dict,
        prediction_time: float = 0.1,
        dt: float = 0.02,  # 时间步长，默认0.02秒
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """使用历史数据预测未来时刻的平台状态（改进版：与训练保持一致）
        
        关键改进：使用与训练一致的方法，确保训练和预测的数据分布一致
        - 训练时：使用延迟历史数据（t-delay_steps之前）预测未来多步
        - 预测时：使用当前时刻之前的历史数据，预测未来prediction_time后的状态
        
        为了与训练保持一致，我们使用"当前时刻之前"的历史数据（不包括当前时刻），
        然后选择对应的预测步数（根据prediction_time计算）。
        
        Args:
            platform_history: 平台历史数据字典，包含：
                - 'quat_w': 历史四元数（可以是tensor stack或列表）[history_length, num_envs, 4]
                - 'ang_vel_w': 历史角速度（可以是tensor stack或列表）[history_length, num_envs, 3]
            prediction_time: 预测时间范围（秒），例如0.1秒表示预测未来0.1秒后的状态
            dt: 时间步长（秒），默认0.02秒
        
        Returns:
            predicted_roll: 预测的未来roll角度 [num_envs]
            predicted_pitch: 预测的未来pitch角度 [num_envs]
            predicted_roll_ang_vel: 预测的未来roll角速度 [num_envs]
            predicted_pitch_ang_vel: 预测的未来pitch角速度 [num_envs]
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
        
        # 计算需要预测的步数（根据prediction_time和dt）
        prediction_steps_needed = int(prediction_time / dt + 0.5)  # 四舍五入到最近的整数步
        prediction_steps_needed = min(prediction_steps_needed, self.prediction_steps)  # 不超过模型能预测的最大步数
        
        # 关键改进：使用"当前时刻之前"的历史数据（不包括当前时刻），与训练保持一致
        # 这样模型学习的是"从历史数据预测未来"，而不是"从当前数据预测未来"
        # 历史数据长度：至少需要history_length，但使用当前时刻之前的数据
        total_history_length = quat_history.shape[0]
        if total_history_length < self.history_length + 1:  # 至少需要history_length+1（包括当前时刻）
            # 历史数据不足，返回零
            num_envs = quat_history.shape[1] if quat_history.shape[0] > 0 else 1
            device = next(self.parameters()).device
            return (
                torch.zeros(num_envs, device=device),
                torch.zeros(num_envs, device=device),
                torch.zeros(num_envs, device=device),
                torch.zeros(num_envs, device=device)
            )
        
        # 获取环境数量（用于广播）
        num_envs = quat_history.shape[1]
        
        # 提取历史数据（使用当前时刻之前的历史数据，不包括当前时刻）
        # 使用最后history_length个历史点（不包括当前时刻）
        history_start_idx = max(0, total_history_length - self.history_length - 1)
        history_end_idx = total_history_length - 1  # 不包括当前时刻（最后一个）
        actual_history_length = history_end_idx - history_start_idx
        
        if actual_history_length < 2:
            # 历史数据不足，返回零
            device = next(self.parameters()).device
            return (
                torch.zeros(num_envs, device=device),
                torch.zeros(num_envs, device=device),
                torch.zeros(num_envs, device=device),
                torch.zeros(num_envs, device=device)
            )
        
        # 关键改进：各环境的运动是完全一致的，所以预测结果只需要预测一个平台的运动
        # 但训练时使用所有环境的数据（增加训练数据量）
        # 预测时只使用环境0的数据，然后广播到所有环境
        recent_quat = quat_history[history_start_idx:history_end_idx, 0:1, :]  # [actual_history_length, 1, 4] 只使用环境0
        recent_ang_vel = ang_vel_history[history_start_idx:history_end_idx, 0:1, :]  # [actual_history_length, 1, 3] 只使用环境0
        
        # 提取roll和pitch角度（只提取环境0）
        history_roll = torch.zeros(actual_history_length, 1, device=recent_quat.device)
        history_pitch = torch.zeros(actual_history_length, 1, device=recent_quat.device)
        
        for i in range(actual_history_length):
            roll_i, pitch_i, _ = euler_xyz_from_quat(recent_quat[i])
            history_roll[i, 0] = roll_i[0]
            history_pitch[i, 0] = pitch_i[0]
        
        # 提取roll和pitch角速度（只提取环境0）
        history_roll_ang_vel = recent_ang_vel[:, 0, 0:1]  # [actual_history_length, 1]
        history_pitch_ang_vel = recent_ang_vel[:, 0, 1:2]  # [actual_history_length, 1]
        
        # 转置为 [1, actual_history_length]
        history_roll = history_roll.transpose(0, 1)  # [1, actual_history_length]
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
        
        # 数据归一化（输入）：与训练保持一致
        history_roll_norm = history_roll * self.input_scale['roll']
        history_pitch_norm = history_pitch * self.input_scale['pitch']
        history_roll_ang_vel_norm = history_roll_ang_vel * self.input_scale['roll_ang_vel']
        history_pitch_ang_vel_norm = history_pitch_ang_vel * self.input_scale['pitch_ang_vel']
        
        # 预测未来多步（使用生产模型，只预测环境0，然后广播到所有环境）
        # 如果生产模型未初始化，使用候选模型（self）
        if self.production_model is None:
            self.eval()
            with torch.no_grad():
                predicted_states_norm = self.forward(
                    history_roll_norm, history_pitch_norm, history_roll_ang_vel_norm, history_pitch_ang_vel_norm
                )  # [1, prediction_steps, 2]
        else:
            self.production_model.eval()
            with torch.no_grad():
                predicted_states_norm = self._forward_with_model(
                    self.production_model,
                    history_roll_norm, history_pitch_norm, history_roll_ang_vel_norm, history_pitch_ang_vel_norm
                )  # [1, prediction_steps, 2]
        
        # 反归一化输出（只包含roll和pitch，2维）
        predicted_states = predicted_states_norm.clone()
        predicted_states[:, :, 0] = predicted_states_norm[:, :, 0] / self.input_scale['roll']
        predicted_states[:, :, 1] = predicted_states_norm[:, :, 1] / self.input_scale['pitch']
        # 注意：不再包含角速度反归一化，因为输出只包含roll和pitch（2维）
        
        # 选择对应的预测步数（根据prediction_time计算）
        # 例如：prediction_time=0.1秒，dt=0.02秒，需要预测5步，选择step_idx=4（第5步，索引从0开始）
        step_idx = min(prediction_steps_needed - 1, self.prediction_steps - 1)
        
        # 提取对应步数的预测（环境0）
        predicted_state = predicted_states[0, step_idx, :]  # [2] (roll, pitch)
        
        # 广播到所有环境（因为各环境的运动是完全一致的）
        predicted_roll = predicted_state[0].expand(num_envs)
        predicted_pitch = predicted_state[1].expand(num_envs)
        # 角速度暂时设为0（如果需要可以后续添加）
        predicted_roll_ang_vel = torch.zeros_like(predicted_roll)
        predicted_pitch_ang_vel = torch.zeros_like(predicted_roll)
        
        return predicted_roll, predicted_pitch, predicted_roll_ang_vel, predicted_pitch_ang_vel
    
    def predict_future_from_observations(
        self,
        observation_history: dict,
        prediction_steps: int = 5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """使用机器狗观测历史预测未来平台运动
        
        Args:
            observation_history: 机器狗观测历史字典，包含：
                - base_lin_vel: [batch_size, history_length, 3]
                - base_ang_vel: [batch_size, history_length, 3]
                - projected_gravity: [batch_size, history_length, 3]
                - velocity_commands: [batch_size, history_length, num_velocity_commands]
                - joint_pos: [batch_size, history_length, num_joints]
                - joint_vel: [batch_size, history_length, num_joints]
                - actions: [batch_size, history_length, num_actions]
            prediction_steps: 预测未来多少步
        
        Returns:
            predicted_roll: 预测的未来roll角度 [num_envs]
            predicted_pitch: 预测的未来pitch角度 [num_envs]
            predicted_roll_ang_vel: 预测的未来roll角速度 [num_envs]
            predicted_pitch_ang_vel: 预测的未来pitch角速度 [num_envs]
        """
        self.eval()
        with torch.no_grad():
            # ========== 关键修复：恢复归一化，确保与训练时一致 ==========
            # 必须使用与训练时相同的归一化方法，否则预测结果会不准确
            
            # 归一化输入：使用与训练时相同的方法
            def normalize_feature(feature_tensor, feature_dim):
                """归一化特征到[-1, 1]范围（与训练时一致）"""
                mean = feature_tensor.mean(dim=(0, 1), keepdim=True)  # [1, 1, feature_dim]
                std = feature_tensor.std(dim=(0, 1), keepdim=True) + 1e-8  # [1, 1, feature_dim]
                normalized = (feature_tensor - mean) / (std * 3.0)
                return torch.clamp(normalized, -1.0, 1.0)
            
            # 归一化所有输入特征
            obs_norm = {
                'base_lin_vel': normalize_feature(observation_history['base_lin_vel'], 3),
                'base_ang_vel': normalize_feature(observation_history['base_ang_vel'], 3),
                'projected_gravity': normalize_feature(observation_history['projected_gravity'], 3),
                'velocity_commands': normalize_feature(observation_history['velocity_commands'], observation_history['velocity_commands'].shape[-1]),
                'joint_pos': normalize_feature(observation_history['joint_pos'], observation_history['joint_pos'].shape[-1]),
                'joint_vel': normalize_feature(observation_history['joint_vel'], observation_history['joint_vel'].shape[-1]),
                'actions': normalize_feature(observation_history['actions'], observation_history['actions'].shape[-1]),
            }
            
            # 使用生产模型进行预测（如果已初始化）
            model = self.production_model if (hasattr(self, 'production_model') and self.production_model is not None) else self
            
            # 前向传播（使用归一化的输入）
            if isinstance(model, nn.ModuleDict):
                # 生产模型：使用ModuleDict
                predicted_states_norm = self._forward_with_model(
                    model,
                    obs_norm['base_lin_vel'],
                    obs_norm['base_ang_vel'],
                    obs_norm['projected_gravity'],
                    obs_norm['velocity_commands'],
                    obs_norm['joint_pos'],
                    obs_norm['joint_vel'],
                    obs_norm['actions'],
                )  # [batch_size, prediction_steps, 2] (归一化的)
            else:
                # 候选模型：直接使用self
                predicted_states_norm = self.forward(
                    obs_norm['base_lin_vel'],
                    obs_norm['base_ang_vel'],
                    obs_norm['projected_gravity'],
                    obs_norm['velocity_commands'],
                    obs_norm['joint_pos'],
                    obs_norm['joint_vel'],
                    obs_norm['actions'],
                )  # [batch_size, prediction_steps, 2] (归一化的)
            
            # 关键：预测输出是归一化的，需要反归一化
            # 但是预测时我们没有batch_targets来计算统计值，所以需要估计
            # 简单方法：假设平台状态在合理范围内，使用经验值进行反归一化
            # 更好的方法：使用保存的统计信息（如果有）
            
            # 尝试使用保存的统计信息
            if self.target_stats['mean'] is not None and self.target_stats['std'] is not None:
                device = predicted_states_norm.device
                target_mean = self.target_stats['mean'].to(device)  # [2]
                target_std = self.target_stats['std'].to(device)  # [2]
                # 扩展维度
                target_mean = target_mean.unsqueeze(0).unsqueeze(0)  # [1, 1, 2]
                target_std = target_std.unsqueeze(0).unsqueeze(0)  # [1, 1, 2]
                # 反归一化
                predicted_states = predicted_states_norm * (target_std * 3.0) + target_mean
            else:
                # 如果没有统计信息，假设平台状态在合理范围内
                # roll和pitch通常在-0.5到0.5 rad之间
                # 使用这些经验值进行反归一化
                device = predicted_states_norm.device
                estimated_mean = torch.tensor([0.0, 0.0], device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, 2]
                estimated_std = torch.tensor([0.2, 0.2], device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, 2] (roll, pitch)
                predicted_states = predicted_states_norm * (estimated_std * 3.0) + estimated_mean
            
            # 选择对应的预测步数
            step_idx = min(prediction_steps - 1, predicted_states.shape[1] - 1)
            predicted_state = predicted_states[0, step_idx, :]  # [2] - 只取环境0: [roll, pitch]
            
            # 广播到所有环境
            num_envs = observation_history['base_lin_vel'].shape[0] if observation_history['base_lin_vel'].shape[0] > 1 else 1
            predicted_roll = predicted_state[0].expand(num_envs)
            predicted_pitch = predicted_state[1].expand(num_envs)
            
            # 角速度暂时设为0（如果需要可以后续添加）
            predicted_roll_ang_vel = torch.zeros_like(predicted_roll)
            predicted_pitch_ang_vel = torch.zeros_like(predicted_roll)
        
        return predicted_roll, predicted_pitch, predicted_roll_ang_vel, predicted_pitch_ang_vel
    
    def update_from_observations(
        self,
        observation_history: dict,  # 机器狗观测历史
        future_states: torch.Tensor,  # [batch_size, prediction_steps, 6] - 未来平台状态（6自由度：x, y, z, roll, pitch, yaw）
    ):
        """使用机器狗观测历史更新网络参数（简化版）
        
        Args:
            observation_history: 机器狗观测历史字典，包含：
                - base_lin_vel: [batch_size, history_length, 3]
                - base_ang_vel: [batch_size, history_length, 3]
                - projected_gravity: [batch_size, history_length, 3]
                - velocity_commands: [batch_size, history_length, num_velocity_commands]
                - joint_pos: [batch_size, history_length, num_joints]
                - joint_vel: [batch_size, history_length, num_joints]
                - actions: [batch_size, history_length, num_actions]
            future_states: 未来多步的平台状态 [batch_size, prediction_steps, 2] (roll, pitch)
        """
        # 关键改进：将多环境数据拆分成多个单环境样本，简化训练逻辑
        # 获取batch_size（环境数量）
        batch_size = observation_history['base_lin_vel'].shape[0]
        
        # 如果batch_size > 1，将多环境数据拆分成多个单环境样本
        for env_idx in range(batch_size):
            # 为每个环境创建一个单环境样本
            env_obs = (
                observation_history['base_lin_vel'][env_idx:env_idx+1].detach().clone(),  # [1, history_length, 3]
                observation_history['base_ang_vel'][env_idx:env_idx+1].detach().clone(),  # [1, history_length, 3]
                observation_history['projected_gravity'][env_idx:env_idx+1].detach().clone(),  # [1, history_length, 3]
                observation_history['velocity_commands'][env_idx:env_idx+1].detach().clone(),  # [1, history_length, num_velocity_commands]
                observation_history['joint_pos'][env_idx:env_idx+1].detach().clone(),  # [1, history_length, num_joints]
                observation_history['joint_vel'][env_idx:env_idx+1].detach().clone(),  # [1, history_length, num_joints]
                observation_history['actions'][env_idx:env_idx+1].detach().clone(),  # [1, history_length, num_actions]
            )
            env_target = future_states[env_idx:env_idx+1].detach().clone()  # [1, prediction_steps, 2]
            
            # 添加到训练缓冲区
            self.training_buffer['inputs'].append(env_obs)
            self.training_buffer['targets'].append(env_target)
        
        # 限制缓冲区大小
        while len(self.training_buffer['inputs']) > self.training_buffer['max_buffer_size']:
            self.training_buffer['inputs'].pop(0)
            self.training_buffer['targets'].pop(0)
        
        self.train_step_count += 1
        
        # ========== 改进的训练策略：提高训练频率和质量 ==========
        # 目标：让网络更快学习平台运动规律，提高预测准确性
        
        min_batch_size = 64  # 最小批次大小，平衡训练效果和速度
        if len(self.training_buffer['inputs']) >= min_batch_size:
            # 关键改进：平衡训练频率和速度
            # 早期：适度训练，快速学习基础模式
            # 中期：保持训练，继续学习
            # 后期：降低频率，稳定训练
            # 关键优化：大幅降低训练频率，提升训练速度（从10-50步改为100-500步）
            if self.candidate_train_steps < 500:
                train_interval = 100  # 早期：每100步训练一次（从10增加到100，降低10倍）
                num_batches_per_step = 1  # 每次训练1个批次
            elif self.candidate_train_steps < 2000:
                train_interval = 200  # 中期：每200步训练一次（从20增加到200，降低10倍）
                num_batches_per_step = 1  # 每次训练1个批次
            else:
                train_interval = 500  # 后期：每500步训练一次（从50增加到500，降低10倍）
                num_batches_per_step = 1  # 每次训练1个批次
            
            if self.train_step_count % train_interval == 0:
                # 改进2：根据缓冲区大小动态调整批次数量
                # 缓冲区越大，每次训练更多批次，充分利用数据
                buffer_size = len(self.training_buffer['inputs'])
                # 提高阈值，减少训练频率，但保持学习率修复的效果
                dynamic_batches = max(1, min(num_batches_per_step, buffer_size // 1000))
                
                for _ in range(dynamic_batches):
                    self._train_batch_from_observations()
    
    def _train_batch_from_observations(self):
        """使用机器狗观测历史训练一个批次（增强版，支持更好的训练）"""
        if len(self.training_buffer['inputs']) < 64:  # 降低最小批次大小要求，更早开始训练
            return
        
        # 动态批次大小：根据训练步数逐渐增加，提高训练稳定性
        base_batch_size = 128  # 从64增加到128（增加批次大小，提高训练稳定性和收敛速度）
        if self.candidate_train_steps < 1000:
            batch_size = base_batch_size  # 早期：64
        elif self.candidate_train_steps < 5000:
            batch_size = min(128, base_batch_size * 2)  # 中期：128
        else:
            batch_size = min(256, base_batch_size * 4)  # 后期：256
        
        batch_size = min(batch_size, len(self.training_buffer['inputs']))
        
        # 关键改进：使用困难样本挖掘策略，优先训练预测误差大的样本
        # 这样可以更快地学习困难案例，提高训练效率
        if len(self.training_buffer['inputs']) > batch_size and hasattr(self, 'hard_sample_buffer') and len(self.hard_sample_buffer['inputs']) > 0:
            # 混合采样：70%来自困难样本，30%来自随机样本
            num_hard_samples = int(batch_size * 0.7)
            num_random_samples = batch_size - num_hard_samples
            
            # 从困难样本缓冲区采样
            hard_indices = torch.randint(0, len(self.hard_sample_buffer['inputs']), 
                                        (min(num_hard_samples, len(self.hard_sample_buffer['inputs'])),), 
                                        device=self.device)
            # 从普通缓冲区随机采样
            random_indices = torch.randint(0, len(self.training_buffer['inputs']), 
                                          (num_random_samples,), 
                                          device=self.device)
            # 标记哪些是困难样本（用于后续处理）
            indices = (hard_indices, random_indices, True)  # (hard_indices, random_indices, is_mixed)
        else:
            # 如果困难样本缓冲区为空，使用随机采样
            indices = (torch.randint(0, len(self.training_buffer['inputs']), (batch_size,), device=self.device), None, False)
        
        # 准备批量数据
        batch_obs = {
            'base_lin_vel': [],
            'base_ang_vel': [],
            'projected_gravity': [],
            'velocity_commands': [],
            'joint_pos': [],
            'joint_vel': [],
            'actions': [],
        }
        batch_targets = []
        
        # 处理混合采样或随机采样
        # 关键修复：保存all_indices以便后续使用
        if isinstance(indices, tuple) and indices[2]:  # 混合采样
            hard_indices, random_indices, _ = indices
            all_indices = []
            # 从困难样本缓冲区获取样本
            for idx in hard_indices:
                all_indices.append(('hard', idx.item()))
            # 从普通缓冲区获取样本
            for idx in random_indices:
                all_indices.append(('normal', idx.item()))
        else:  # 随机采样
            random_indices = indices[0] if isinstance(indices, tuple) else indices
            all_indices = [('normal', idx.item()) for idx in random_indices]
        
        # 保存all_indices到self，以便后续更新困难样本缓冲区时使用
        self._current_batch_indices = all_indices
        
        for sample_type, idx in all_indices:
            if sample_type == 'hard':
                # 从困难样本缓冲区获取
                obs = self.hard_sample_buffer['inputs'][idx]
                target = self.hard_sample_buffer['targets'][idx]
            else:
                # 从普通缓冲区获取
                obs = self.training_buffer['inputs'][idx]
                target = self.training_buffer['targets'][idx]
            
            # obs是一个元组，包含7个元素（已移除height_scan）
            # 每个元素应该是 [1, history_length, ...] 或 [history_length, ...] 的形状
            # 因为update_from_observations已经将多环境数据拆分成单环境样本
            def ensure_2d(tensor):
                """确保张量是2D的 [history_length, feature_dim]"""
                if tensor.dim() == 3:
                    if tensor.shape[0] == 1:
                        return tensor.squeeze(0)  # [1, history_length, ...] -> [history_length, ...]
                    else:
                        raise ValueError(f"Unexpected tensor shape: {tensor.shape}. Expected [1, history_length, ...] or [history_length, ...]")
                elif tensor.dim() == 2:
                    return tensor  # 已经是 [history_length, ...]
                else:
                    raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
            
            # 处理观测数据（单环境数据）
            batch_obs['base_lin_vel'].append(ensure_2d(obs[0]))  # [history_length, 3]
            batch_obs['base_ang_vel'].append(ensure_2d(obs[1]))  # [history_length, 3]
            batch_obs['projected_gravity'].append(ensure_2d(obs[2]))  # [history_length, 3]
            batch_obs['velocity_commands'].append(ensure_2d(obs[3]))  # [history_length, num_velocity_commands]
            batch_obs['joint_pos'].append(ensure_2d(obs[4]))  # [history_length, num_joints]
            batch_obs['joint_vel'].append(ensure_2d(obs[5]))  # [history_length, num_joints]
            batch_obs['actions'].append(ensure_2d(obs[6]))  # [history_length, num_actions]
            # 注意：已移除height_scan，obs现在只有7个元素（索引0-6）
            
            # target应该是 [1, prediction_steps, 6] 或 [prediction_steps, 6]
            if target.dim() == 3 and target.shape[0] == 1:
                batch_targets.append(target.squeeze(0))  # [prediction_steps, 4]
            elif target.dim() == 2:
                batch_targets.append(target)  # [prediction_steps, 4]
            else:
                raise ValueError(f"Unexpected target shape: {target.shape}. Expected [1, prediction_steps, 4] or [prediction_steps, 4]")
        
        # 堆叠为批量
        # 关键：在梯度上下文中创建batch_targets，确保它不是inference tensor
        with torch.enable_grad(), torch.inference_mode(False):
            for key in batch_obs.keys():
                batch_obs[key] = torch.stack(batch_obs[key], dim=0)  # [batch_size, history_length, ...]
            batch_targets = torch.stack(batch_targets, dim=0)  # [batch_size, prediction_steps, 4]
        
        # 确保数据在正确的设备上
        # 关键：使用.clone()创建普通张量，避免inference tensor问题
        device = next(self.parameters()).device
        for key in batch_obs.keys():
            batch_obs[key] = batch_obs[key].clone().to(device)
        # batch_targets需要是普通张量（不是inference tensor），才能用于损失计算
        batch_targets = batch_targets.clone().to(device)
        
        # 训练模式
        self.train()
        
        # 确保所有参数需要梯度
        for param in self.parameters():
            param.requires_grad = True
        
        # 确保优化器包含所有参数
        if len(self.optimizer.param_groups) == 0 or len(self.optimizer.param_groups[0]['params']) != len(list(self.parameters())):
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate,
                weight_decay=1e-5,
            )
        
        self.optimizer.zero_grad()
        
        # 前向传播和损失计算（在梯度启用上下文中）
        # 关键：使用torch.enable_grad()和torch.inference_mode(False)确保梯度计算
        with torch.enable_grad(), torch.inference_mode(False):
            # 确保模型处于训练模式
            self.train()
            
            # ========== 关键修复：恢复归一化，确保网络能正确学习 ==========
            # 问题：不同特征的量纲差异很大（关节位置0-1，角速度-10到10），不归一化会导致训练困难
            # 解决方案：使用稳定的归一化方法，确保训练和预测一致
            
            # 归一化输入：使用batch内的统计值（简单但有效）
            def normalize_feature(feature_tensor, feature_dim):
                """归一化特征到[-1, 1]范围"""
                # 计算每个特征的均值和标准差
                mean = feature_tensor.mean(dim=(0, 1), keepdim=True)  # [1, 1, feature_dim]
                std = feature_tensor.std(dim=(0, 1), keepdim=True) + 1e-8  # [1, 1, feature_dim]
                # 归一化：使用3倍标准差覆盖99.7%的数据
                normalized = (feature_tensor - mean) / (std * 3.0)
                return torch.clamp(normalized, -1.0, 1.0), mean, std
            
            # 归一化所有输入特征
            batch_obs_norm = {}
            obs_stats = {}
            batch_obs_norm['base_lin_vel'], obs_stats['base_lin_vel_mean'], obs_stats['base_lin_vel_std'] = \
                normalize_feature(batch_obs['base_lin_vel'], 3)
            batch_obs_norm['base_ang_vel'], obs_stats['base_ang_vel_mean'], obs_stats['base_ang_vel_std'] = \
                normalize_feature(batch_obs['base_ang_vel'], 3)
            batch_obs_norm['projected_gravity'], obs_stats['projected_gravity_mean'], obs_stats['projected_gravity_std'] = \
                normalize_feature(batch_obs['projected_gravity'], 3)
            batch_obs_norm['velocity_commands'], obs_stats['velocity_commands_mean'], obs_stats['velocity_commands_std'] = \
                normalize_feature(batch_obs['velocity_commands'], batch_obs['velocity_commands'].shape[-1])
            batch_obs_norm['joint_pos'], obs_stats['joint_pos_mean'], obs_stats['joint_pos_std'] = \
                normalize_feature(batch_obs['joint_pos'], batch_obs['joint_pos'].shape[-1])
            batch_obs_norm['joint_vel'], obs_stats['joint_vel_mean'], obs_stats['joint_vel_std'] = \
                normalize_feature(batch_obs['joint_vel'], batch_obs['joint_vel'].shape[-1])
            batch_obs_norm['actions'], obs_stats['actions_mean'], obs_stats['actions_std'] = \
                normalize_feature(batch_obs['actions'], batch_obs['actions'].shape[-1])
            
            # 归一化目标（平台状态）
            target_mean = batch_targets.mean(dim=(0, 1), keepdim=True)  # [1, 1, 4]
            target_std = batch_targets.std(dim=(0, 1), keepdim=True) + 1e-8  # [1, 1, 4]
            batch_targets_norm = (batch_targets - target_mean) / (target_std * 3.0)
            batch_targets_norm = torch.clamp(batch_targets_norm, -1.0, 1.0)
            
            # 前向传播（使用归一化的输入）
            predicted_norm = self.forward(
                batch_obs_norm['base_lin_vel'],
                batch_obs_norm['base_ang_vel'],
                batch_obs_norm['projected_gravity'],
                batch_obs_norm['velocity_commands'],
                batch_obs_norm['joint_pos'],
                batch_obs_norm['joint_vel'],
                batch_obs_norm['actions'],
            )  # [batch_size, prediction_steps, 4] (归一化的)
            
            # 反归一化预测结果（用于损失计算）
            predicted = predicted_norm * (target_std * 3.0) + target_mean
            
            # ========== 关键：更新归一化统计信息（用于预测时的反归一化） ==========
            # 使用指数移动平均更新统计信息，确保预测时可以使用
            alpha = 0.01  # 更新率
            for key in batch_obs.keys():
                obs_mean = batch_obs[key].mean(dim=(0, 1), keepdim=False)  # [feature_dim]
                obs_std = batch_obs[key].std(dim=(0, 1), keepdim=False) + 1e-8  # [feature_dim]
                if self.obs_stats[key]['mean'] is None:
                    self.obs_stats[key]['mean'] = obs_mean.detach().clone()
                    self.obs_stats[key]['std'] = obs_std.detach().clone()
                else:
                    self.obs_stats[key]['mean'] = (1 - alpha) * self.obs_stats[key]['mean'] + alpha * obs_mean.detach()
                    self.obs_stats[key]['std'] = (1 - alpha) * self.obs_stats[key]['std'] + alpha * obs_std.detach()
                self.obs_stats[key]['count'] += 1
            
            # 更新目标统计信息
            target_mean_for_update = batch_targets.mean(dim=(0, 1), keepdim=False)  # [4]
            target_std_for_update = batch_targets.std(dim=(0, 1), keepdim=False) + 1e-8  # [4]
            if self.target_stats['mean'] is None:
                self.target_stats['mean'] = target_mean_for_update.detach().clone()
                self.target_stats['std'] = target_std_for_update.detach().clone()
            else:
                self.target_stats['mean'] = (1 - alpha) * self.target_stats['mean'] + alpha * target_mean_for_update.detach()
                self.target_stats['std'] = (1 - alpha) * self.target_stats['std'] + alpha * target_std_for_update.detach()
            self.target_stats['count'] += 1
            
            # 检查预测是否有梯度
            if not predicted.requires_grad:
                # 如果预测没有梯度，说明计算图被破坏了
                print(f"[警告] 预测结果没有梯度！检查模型状态...")
                print(f"  - 模型training模式: {self.training}")
                print(f"  - 参数需要梯度数量: {sum(1 for p in self.parameters() if p.requires_grad)}/{len(list(self.parameters()))}")
                print(f"  - 梯度启用: {torch.is_grad_enabled()}")
                print(f"  - predicted.grad_fn: {predicted.grad_fn}")
                
                # 强制重新设置
                for param in self.parameters():
                    param.requires_grad = True
                self.train()
                
                # 重新计算
                predicted = self.forward(
                    batch_obs['base_lin_vel'],
                    batch_obs['base_ang_vel'],
                    batch_obs['projected_gravity'],
                    batch_obs['velocity_commands'],
                    batch_obs['joint_pos'],
                    batch_obs['joint_vel'],
                    batch_obs['actions'],
                )
                
                if not predicted.requires_grad:
                    print(f"[错误] 重新计算后仍然没有梯度，跳过此次训练")
                    return
            
            # 计算损失（在同一个上下文中）
            # 关键：确保batch_targets不是inference tensor，使用.clone()创建普通张量
            batch_targets_normal = batch_targets.clone() if batch_targets.is_inference() else batch_targets
            mse_loss = self.criterion_mse(predicted, batch_targets_normal)
            huber_loss = self.criterion_huber(predicted, batch_targets_normal)
            
            # ========== 改进的损失函数：更好地学习平台运动规律 ==========
            # 目标：让网络学习平台运动的周期性、趋势和变化率
            
            # 1. 基础损失：姿态的加权组合（只预测roll和pitch，2维）
            weights = torch.ones_like(mse_loss)
            weights[:, :, 0] = 3.0  # roll权重（增加，因为roll更重要）
            weights[:, :, 1] = 3.0  # pitch权重（增加，因为pitch更重要）
            # 注意：不再包含角速度权重，因为输出只包含roll和pitch（2维）
            
            # 2. 时间权重：越近的预测步权重越大（更关注短期预测）
            for step_idx in range(self.prediction_steps):
                time_weight = 1.0 + (self.prediction_steps - step_idx) * 0.2  # 第一预测步权重最大
                weights[:, step_idx, :] *= time_weight
            
            # 3. 关键改进：使用Focal Loss思想，重点关注困难样本（误差大的样本）
            # Focal Loss通过降低简单样本的权重，让模型更关注困难样本
            error_magnitude = torch.abs(predicted - batch_targets)  # [batch_size, prediction_steps, 2]
            # 计算每个样本的困难程度（误差越大越困难）
            sample_difficulty = error_magnitude.mean(dim=(1, 2))  # [batch_size]
            # Focal权重：困难样本权重更高
            focal_alpha = 0.25  # 平衡因子
            focal_gamma = 2.0   # 聚焦参数（gamma越大，越关注困难样本）
            # 归一化困难程度到[0, 1]
            normalized_difficulty = (sample_difficulty - sample_difficulty.min()) / (sample_difficulty.max() - sample_difficulty.min() + 1e-8)
            # Focal权重：困难样本权重 = alpha * (1 - normalized_difficulty)^gamma
            # 但这里我们反过来：困难样本权重更高
            focal_weights = focal_alpha + (1 - focal_alpha) * (normalized_difficulty ** focal_gamma)
            focal_weights = focal_weights.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1]
            
            # 使用Huber Loss处理小误差，MSE Loss处理大误差
            elementwise_loss = torch.where(
                error_magnitude < 0.05,  # 小误差使用Huber Loss
                huber_loss,
                mse_loss
            )
            # 应用Focal权重
            elementwise_loss = elementwise_loss * focal_weights
            
            # 4. 添加姿态趋势损失：鼓励网络学习姿态的变化趋势
            # 计算预测的姿态变化率（一阶差分）
            if self.prediction_steps > 1:
                predicted_orientation_change = predicted[:, 1:, :] - predicted[:, :-1, :]  # [batch_size, prediction_steps-1, 2] (roll, pitch)
                target_orientation_change = batch_targets[:, 1:, :] - batch_targets[:, :-1, :]  # [batch_size, prediction_steps-1, 2] (roll, pitch)
                trend_loss = torch.mean((predicted_orientation_change - target_orientation_change) ** 2)
            else:
                trend_loss = torch.tensor(0.0, device=predicted.device)
            
            # 5. 组合损失：基础损失 + 趋势损失
            base_loss = (elementwise_loss * weights).mean()
            loss = base_loss + 0.1 * trend_loss  # 趋势损失权重较小，作为辅助
            
            # 检查损失是否有梯度
            if not loss.requires_grad:
                print(f"[错误] 损失没有梯度，无法反向传播")
                print(f"  - loss.grad_fn: {loss.grad_fn}")
                print(f"  - predicted.requires_grad: {predicted.requires_grad}")
                return
            
            # 反向传播
            loss.backward()
        
        
        # 梯度裁剪（动态调整）
        max_norm = 1.0
        if self.candidate_train_steps > 10000:
            max_norm = 0.5  # 后期使用更小的梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=max_norm)
        
        # 更新参数
        self.optimizer.step()
        self.candidate_train_steps += 1
        
        # 更新学习率（支持长期训练和跳出局部最优）
        if not hasattr(self, '_scheduler_initialized') or not self._scheduler_initialized:
            # 使用CosineAnnealingWarmRestarts，支持周期性重启
            # 关键修复：提高最小学习率，防止学习率变成0
            min_lr = max(self.learning_rate * 0.1, 1e-5)  # 最小学习率：learning_rate的10%，但不少于1e-5
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=2000,  # 初始周期：2000步
                T_mult=1,  # 周期倍增因子（1表示周期长度不变）
                eta_min=min_lr,  # 最小学习率（修复：从0.001提高到0.1）
            )
            self._scheduler_initialized = True
        else:
            self.scheduler.step()
        
        # 自适应学习率调整：如果损失长时间不下降，增加学习率
        # 关键修复：添加学习率下限保护，防止学习率变成0
        min_lr_threshold = max(self.learning_rate * 0.1, 1e-5)  # 学习率下限
        max_lr_threshold = self.learning_rate * 2.0  # 学习率上限
        
        if len(self.loss_history) >= 200:
            recent_losses = self.loss_history[-200:]
            avg_recent_loss = sum(recent_losses) / len(recent_losses)
            avg_prev_loss = sum(self.loss_history[-400:-200]) / 200 if len(self.loss_history) >= 400 else avg_recent_loss
            
            loss_improvement = (avg_prev_loss - avg_recent_loss) / (avg_prev_loss + 1e-8)
            
            if loss_improvement < 0.01 and self.candidate_train_steps % 500 == 0:  # 如果200步内改善小于1%
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # 关键修复：如果学习率太低，直接重置到初始学习率
                if current_lr < min_lr_threshold:
                    new_lr = self.learning_rate  # 重置到初始学习率
                else:
                    new_lr = min(current_lr * 1.2, max_lr_threshold)  # 增加学习率，但不超过上限
                
                # 确保学习率在合理范围内
                new_lr = max(new_lr, min_lr_threshold)
                new_lr = min(new_lr, max_lr_threshold)
                
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                print(f"[平台预测器] 步骤 {self.candidate_train_steps}: 损失改善过慢 ({loss_improvement:.4f})，调整学习率从 {current_lr:.6f} 到 {new_lr:.6f}")
        
        # 关键修复：每步都检查学习率，确保不会太小
        current_lr = self.optimizer.param_groups[0]['lr']
        if current_lr < min_lr_threshold:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = min_lr_threshold
        
        # 关键改进：更新困难样本缓冲区
        # 将预测误差大的样本加入困难样本缓冲区，用于后续训练
        with torch.no_grad():
            # 计算每个样本的预测误差
            sample_errors = torch.abs(predicted - batch_targets).mean(dim=(1, 2))  # [batch_size]
            # 选择误差最大的30%作为困难样本
            num_hard_samples = max(1, int(batch_size * 0.3))
            _, hard_indices_in_batch = torch.topk(sample_errors, num_hard_samples, largest=True)
            
            # 将困难样本添加到缓冲区
            # 关键修复：使用all_indices来获取原始缓冲区索引
            if hasattr(self, '_current_batch_indices') and len(self._current_batch_indices) == batch_size:
                for batch_idx in hard_indices_in_batch:
                    batch_idx_item = batch_idx.item()
                    if batch_idx_item < len(self._current_batch_indices):
                        sample_type, original_idx = self._current_batch_indices[batch_idx_item]
                        
                        # 根据样本类型获取观测和目标
                        if sample_type == 'hard':
                            # 从困难样本缓冲区获取（这些样本已经在困难缓冲区中，不需要重复添加）
                            # 但我们可以更新它们的误差值
                            if original_idx < len(self.hard_sample_buffer['inputs']):
                                # 更新误差值（如果新误差更大）
                                if sample_errors[batch_idx_item].item() > self.hard_sample_buffer['errors'][original_idx]:
                                    self.hard_sample_buffer['errors'][original_idx] = sample_errors[batch_idx_item].item()
                        else:
                            # 从普通缓冲区获取，添加到困难样本缓冲区
                            if original_idx < len(self.training_buffer['inputs']):
                                hard_obs = self.training_buffer['inputs'][original_idx]
                                hard_target = self.training_buffer['targets'][original_idx]
                                
                                # 添加到困难样本缓冲区
                                self.hard_sample_buffer['inputs'].append(hard_obs)
                                self.hard_sample_buffer['targets'].append(hard_target)
                                self.hard_sample_buffer['errors'].append(sample_errors[batch_idx_item].item())
                                
                                # 限制缓冲区大小
                                if len(self.hard_sample_buffer['inputs']) > self.hard_sample_buffer['max_size']:
                                    # 移除误差最小的样本
                                    min_error_idx = min(range(len(self.hard_sample_buffer['errors'])), 
                                                      key=lambda i: self.hard_sample_buffer['errors'][i])
                                    self.hard_sample_buffer['inputs'].pop(min_error_idx)
                                    self.hard_sample_buffer['targets'].pop(min_error_idx)
                                    self.hard_sample_buffer['errors'].pop(min_error_idx)
            
            # 清理临时变量
            if hasattr(self, '_current_batch_indices'):
                del self._current_batch_indices
        
        # 记录损失历史
        if len(self.loss_history) >= 500:  # 增大历史窗口
            self.loss_history.pop(0)
        self.loss_history.append(loss.item())
        
        # 定期打印训练信息
        if self.candidate_train_steps % 500 == 0:
            current_lr = self.optimizer.param_groups[0]['lr']
            avg_loss = sum(self.loss_history[-100:]) / min(100, len(self.loss_history))
            # 计算真实损失（反归一化后的误差）
            real_error = torch.abs(predicted - batch_targets).mean().item()
            print(f"[平台预测器训练] 步骤 {self.candidate_train_steps}: "
                  f"损失={loss.item():.6f}, 平均损失={avg_loss:.6f}, "
                  f"真实误差={real_error:.6f} rad, "
                  f"学习率={current_lr:.6f}, 缓冲区大小={len(self.training_buffer['inputs'])}")
    
    def update(self, platform_history: dict, 
               actual_roll: torch.Tensor, actual_pitch: torch.Tensor,
               actual_roll_ang_vel: torch.Tensor, actual_pitch_ang_vel: torch.Tensor,
               future_states: torch.Tensor | None = None,
               current_time: float | None = None):
        """使用实际平台状态更新网络参数（在线学习）
        
        Args:
            platform_history: 平台历史数据（用于预测，可以是列表或tensor）
            actual_roll: 实际roll角度 [num_envs]（当前时刻，用于兼容性）
            actual_pitch: 实际pitch角度 [num_envs]（当前时刻，用于兼容性）
            actual_roll_ang_vel: 实际roll角速度 [num_envs]（当前时刻，用于兼容性）
            actual_pitch_ang_vel: 实际pitch角速度 [num_envs]（当前时刻，用于兼容性）
            future_states: 未来多步的实际状态 [num_envs, prediction_steps, 4]（可选）
                如果提供，使用多步预测进行训练；否则只使用当前状态
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
        
        # 提取历史数据（使用所有环境的数据进行训练，因为所有环境的平台运动是一致的）
        # 这样可以大大增加训练数据量，提高训练效率
        recent_quat = quat_history[-actual_history_length:]  # [actual_history_length, num_envs, 4]
        recent_ang_vel = ang_vel_history[-actual_history_length:]  # [actual_history_length, num_envs, 3]
        
        num_envs = recent_quat.shape[1]
        
        # 提取roll和pitch角度（所有环境）
        history_roll = torch.zeros(actual_history_length, num_envs, device=recent_quat.device)
        history_pitch = torch.zeros(actual_history_length, num_envs, device=recent_quat.device)
        
        for i in range(actual_history_length):
            roll_i, pitch_i, _ = euler_xyz_from_quat(recent_quat[i])
            history_roll[i] = roll_i
            history_pitch[i] = pitch_i
        
        # 提取roll和pitch角速度（所有环境）
        history_roll_ang_vel = recent_ang_vel[:, :, 0]  # [actual_history_length, num_envs]
        history_pitch_ang_vel = recent_ang_vel[:, :, 1]  # [actual_history_length, num_envs]
        
        # 转置为 [num_envs, actual_history_length]
        history_roll = history_roll.transpose(0, 1)  # [num_envs, actual_history_length]
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
        
        # 准备输入和目标（使用所有环境的数据进行训练）
        # 因为所有环境的平台运动是一致的，所以可以使用所有环境的数据来训练同一个网络
        # 这样可以大大增加训练数据量，提高训练效率
        # 为所有环境准备目标数据
        if future_states is not None:
            # 如果提供了未来多步状态，使用多步预测进行训练
            # future_states 形状: [num_envs, prediction_steps, 2] (roll, pitch)
            all_targets = future_states  # [num_envs, prediction_steps, 2]
        else:
            # 否则只使用当前状态（为了兼容性，但会重复prediction_steps次）
            current_state = torch.stack([
                actual_roll,  # [num_envs]
                actual_pitch,  # [num_envs]
            ], dim=1)  # [num_envs, 2]
            # 重复prediction_steps次，形成 [num_envs, prediction_steps, 2]
            all_targets = current_state.unsqueeze(1).expand(-1, self.prediction_steps, -1)
        
        # 将所有环境的数据添加到训练缓冲区
        # 每个环境的数据作为一个独立的训练样本
        # 注意：只使用roll和pitch历史，不包含角速度（因为预测器只预测roll和pitch）
        for env_idx in range(num_envs):
            env_inputs = (
                history_roll[env_idx:env_idx+1],  # [1, history_length]
                history_pitch[env_idx:env_idx+1],
                history_roll_ang_vel[env_idx:env_idx+1],  # 保留角速度作为输入特征（帮助预测）
                history_pitch_ang_vel[env_idx:env_idx+1]
            )
            env_targets = all_targets[env_idx:env_idx+1]  # [1, prediction_steps, 2] (roll, pitch)
            
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
        
        # 关键改进：自适应归一化 - 根据实际数据范围动态调整归一化参数
        if self.adaptive_normalization_enabled and self.train_step_count % self.stats_update_frequency == 0:
            # 更新数据统计信息（只更新roll和pitch，因为输出只包含这两个）
            current_roll_max = torch.abs(history_roll).max().item()
            current_pitch_max = torch.abs(history_pitch).max().item()
            
            # 使用指数移动平均更新最大值（保留历史信息，避免突然变化）
            alpha = 0.1  # 更新系数
            self.data_stats['roll']['max_abs'] = (1 - alpha) * self.data_stats['roll']['max_abs'] + alpha * current_roll_max
            self.data_stats['pitch']['max_abs'] = (1 - alpha) * self.data_stats['pitch']['max_abs'] + alpha * current_pitch_max
            
            # 更新归一化参数（使用最大值的1.2倍作为范围，留出余量）
            safety_factor = 1.2
            if self.data_stats['roll']['max_abs'] > 0.01:  # 只有当数据足够大时才更新
                self.input_scale['roll'] = 1.0 / (self.data_stats['roll']['max_abs'] * safety_factor)
                self.output_scale['roll'] = self.data_stats['roll']['max_abs'] * safety_factor
            if self.data_stats['pitch']['max_abs'] > 0.01:
                self.input_scale['pitch'] = 1.0 / (self.data_stats['pitch']['max_abs'] * safety_factor)
                self.output_scale['pitch'] = self.data_stats['pitch']['max_abs'] * safety_factor
            # 注意：roll_ang_vel和pitch_ang_vel仍然作为输入特征（帮助预测），但输出只包含roll和pitch
            # 因此不需要更新它们的output_scale，但input_scale可以保留（如果存在）
        
        # 关键调试：确保数据真的被添加到缓冲区
        if self.train_step_count % 100 == 0:
            print(f"[神经网络数据收集] 步数: {self.train_step_count}, 缓冲区大小: {len(self.training_buffer['inputs'])}, "
                  f"环境数: {num_envs}, 训练步数: {self.candidate_train_steps}")
            if self.adaptive_normalization_enabled:
                print(f"[自适应归一化] roll范围: {self.data_stats['roll']['max_abs']:.4f}, "
                      f"pitch范围: {self.data_stats['pitch']['max_abs']:.4f}")
        
        # 每步都尝试训练（如果缓冲区有足够样本）
        # 优化：降低训练频率，减少计算开销，但确保训练真的执行
        if len(self.training_buffer['inputs']) >= self.batch_size:
            # 关键修复：确保训练真的被执行，添加详细调试信息
            try:
                # 关键调试：在训练前打印信息
                if self.train_step_count % 100 == 0:
                    print(f"[神经网络训练调试] 准备训练: 缓冲区大小={len(self.training_buffer['inputs'])}, batch_size={self.batch_size}, 当前训练步数={self.candidate_train_steps}")
                self._train_batch()
                # 关键调试：训练后检查训练步数是否增加
                if self.train_step_count % 100 == 0 and self.candidate_train_steps == 0:
                    print(f"[神经网络训练警告] 训练后训练步数仍为0，可能训练提前返回了")
            except Exception as e:
                # 如果训练出错，打印错误信息，但不中断程序
                print(f"[神经网络训练错误] 训练异常: {e}")
                import traceback
                traceback.print_exc()
    
    def _train_batch(self):
        """批量训练网络（改进版：使用困难样本挖掘和数据增强）"""
        # 关键调试：在方法开始时打印信息
        if self.train_step_count % 100 == 0:
            print(f"[神经网络训练调试] _train_batch被调用: 缓冲区大小={len(self.training_buffer['inputs'])}, batch_size={self.batch_size}")
        
        if len(self.training_buffer['inputs']) < self.batch_size:
            # 关键调试：如果缓冲区样本不足，打印信息
            if self.train_step_count % 200 == 0:
                print(f"[神经网络训练] 缓冲区样本不足: {len(self.training_buffer['inputs'])} < {self.batch_size}，跳过训练")
            return
        
        # 困难样本挖掘：从困难样本缓冲区和普通缓冲区中采样
        num_hard_samples = int(self.batch_size * self.hard_sample_ratio)
        num_normal_samples = self.batch_size - num_hard_samples
        
        # 从困难样本缓冲区采样
        hard_indices = []
        if len(self.hard_sample_buffer['inputs']) > 0 and num_hard_samples > 0:
            hard_buffer_size = len(self.hard_sample_buffer['inputs'])
            hard_indices = torch.randint(0, hard_buffer_size, (min(num_hard_samples, hard_buffer_size),), device=self.device)
        
        # 从普通缓冲区采样
        normal_buffer_size = len(self.training_buffer['inputs'])
        normal_indices = torch.randint(0, normal_buffer_size, (num_normal_samples,), device=self.device)
        
        # 合并索引
        all_indices = []
        if len(hard_indices) > 0:
            # 从困难样本缓冲区获取样本
            for idx in hard_indices:
                all_indices.append(('hard', idx.item()))
        # 从普通缓冲区获取样本
        for idx in normal_indices:
            all_indices.append(('normal', idx.item()))
        
        # 如果困难样本不足，从普通缓冲区补充
        while len(all_indices) < self.batch_size:
            idx = torch.randint(0, normal_buffer_size, (1,), device=self.device).item()
            all_indices.append(('normal', idx))
        
        # 准备批量数据（从困难样本和普通样本中采样）
        batch_inputs_list = []
        batch_targets_list = []
        
        for sample_type, idx in all_indices:
            if sample_type == 'hard':
                inputs = self.hard_sample_buffer['inputs'][idx]
                targets = self.hard_sample_buffer['targets'][idx]
            else:
                inputs = self.training_buffer['inputs'][idx]
                targets = self.training_buffer['targets'][idx]
            
            batch_inputs_list.append(inputs)
            batch_targets_list.append(targets)
        
        # 合并批量数据
        batch_history_roll = torch.cat([inp[0] for inp in batch_inputs_list], dim=0)  # [batch_size, history_length]
        batch_history_pitch = torch.cat([inp[1] for inp in batch_inputs_list], dim=0)
        batch_history_roll_ang_vel = torch.cat([inp[2] for inp in batch_inputs_list], dim=0)
        batch_history_pitch_ang_vel = torch.cat([inp[3] for inp in batch_inputs_list], dim=0)
        batch_targets = torch.cat(batch_targets_list, dim=0)  # [batch_size, prediction_steps, 4]
        
        # 数据增强：添加小量噪声（仅在训练时）
        if self.use_data_augmentation and self.training:
            noise_roll = torch.randn_like(batch_history_roll) * self.noise_std
            noise_pitch = torch.randn_like(batch_history_pitch) * self.noise_std
            noise_roll_vel = torch.randn_like(batch_history_roll_ang_vel) * self.noise_std
            noise_pitch_vel = torch.randn_like(batch_history_pitch_ang_vel) * self.noise_std
            
            batch_history_roll = batch_history_roll + noise_roll
            batch_history_pitch = batch_history_pitch + noise_pitch
            batch_history_roll_ang_vel = batch_history_roll_ang_vel + noise_roll_vel
            batch_history_pitch_ang_vel = batch_history_pitch_ang_vel + noise_pitch_vel
        
        # 确保数据在正确的设备上，并detach（输入数据不需要梯度）
        batch_history_roll = batch_history_roll.detach().to(self.device)
        batch_history_pitch = batch_history_pitch.detach().to(self.device)
        batch_history_roll_ang_vel = batch_history_roll_ang_vel.detach().to(self.device)
        batch_history_pitch_ang_vel = batch_history_pitch_ang_vel.detach().to(self.device)
        batch_targets = batch_targets.detach().to(self.device)
        
        batch_inputs = (
            batch_history_roll,
            batch_history_pitch,
            batch_history_roll_ang_vel,
            batch_history_pitch_ang_vel
        )
        
        # 训练模式（使用候选模型，即self）
        # 关键修复：确保模型处于训练模式，并检查状态
        self.train()
        if not self.training:
            # 如果模型仍然不是训练模式，强制设置
            for module in self.modules():
                if hasattr(module, 'training'):
                    module.training = True
        
        self.optimizer.zero_grad()
        
        # 确保网络参数需要梯度（关键修复：在训练前强制检查并修复）
        params_need_grad_before = sum(1 for p in self.parameters() if p.requires_grad)
        total_params = len(list(self.parameters()))
        
        # 强制设置所有参数需要梯度
        for param in self.parameters():
            param.requires_grad = True
        
        params_need_grad_after = sum(1 for p in self.parameters() if p.requires_grad)
        
        # 关键调试：如果参数之前不需要梯度，打印警告
        if params_need_grad_before != total_params and self.train_step_count % 100 == 0:
            print(f"[神经网络训练修复] 参数梯度状态已修复：{params_need_grad_before}/{total_params} -> {params_need_grad_after}/{total_params}")
        
        # 关键修复：确保优化器包含所有参数
        # 如果参数之前不需要梯度，优化器可能没有包含这些参数
        if params_need_grad_before != total_params:
            # 重新创建优化器，包含所有参数
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.learning_rate
            )
            if self.train_step_count % 100 == 0:
                print(f"[神经网络训练修复] 已重新创建优化器，包含所有 {params_need_grad_after} 个参数")
        
        # 数据归一化（输入）：提高训练稳定性
        # 归一化输入数据，使不同特征的数值范围一致
        batch_history_roll_norm = batch_history_roll * self.input_scale['roll']
        batch_history_pitch_norm = batch_history_pitch * self.input_scale['pitch']
        batch_history_roll_ang_vel_norm = batch_history_roll_ang_vel * self.input_scale['roll_ang_vel']
        batch_history_pitch_ang_vel_norm = batch_history_pitch_ang_vel * self.input_scale['pitch_ang_vel']
        
        # 简化：只计算趋势和时间特征，移除角加速度和曲率（提高计算速度）
        batch_size, history_length = batch_history_roll.shape
        
        # 计算趋势（最近几步的平均变化率）
        batch_history_roll_trend = torch.zeros_like(batch_history_roll)
        batch_history_pitch_trend = torch.zeros_like(batch_history_pitch)
        window_size = min(5, history_length)
        if history_length >= 2:
            for i in range(history_length):
                start_idx = max(0, i - window_size + 1)
                if i > start_idx:
                    window_len = i - start_idx + 1e-8
                    batch_history_roll_trend[:, i] = (batch_history_roll[:, i] - batch_history_roll[:, start_idx]) / window_len
                    batch_history_pitch_trend[:, i] = (batch_history_pitch[:, i] - batch_history_pitch[:, start_idx]) / window_len
        
        # 关键改进：时间特征归一化 - 使用sin/cos编码帮助模型学习周期性
        # 归一化时间索引到[0, 1]
        time_indices = torch.arange(history_length, device=batch_history_roll.device, dtype=batch_history_roll.dtype).unsqueeze(0).expand(batch_size, -1)
        normalized_time = time_indices / max(history_length - 1, 1.0)  # [batch_size, history_length], [0, 1]
        
        # 使用sin/cos编码，帮助模型学习周期性
        # 使用多个频率，覆盖不同周期长度
        period_ratios = [1.0, 2.0, 4.0]  # 不同周期的倍数
        time_features_list = []
        for period_ratio in period_ratios:
            phase = 2.0 * torch.pi * normalized_time * period_ratio
            time_features_list.append(torch.sin(phase))  # [batch_size, history_length]
            time_features_list.append(torch.cos(phase))  # [batch_size, history_length]
        # 组合时间特征：[batch_size, history_length, 6] (3个周期，每个sin+cos)
        batch_history_time = torch.stack(time_features_list, dim=-1)  # [batch_size, history_length, 6]
        
        # 归一化特征
        batch_history_roll_trend_norm = batch_history_roll_trend * self.input_scale['roll_trend']
        batch_history_pitch_trend_norm = batch_history_pitch_trend * self.input_scale['pitch_trend']
        # 时间特征已经在[-1, 1]范围内（sin/cos），不需要额外归一化
        batch_history_time_norm = batch_history_time  # [batch_size, history_length, 6]
        
        # 组合归一化后的输入：保持为元组形式，以便传递给forward方法
        # 注意：forward方法需要分别的参数，不是组合后的张量
        batch_inputs_norm = (
            batch_history_roll_norm,  # [batch_size, history_length]
            batch_history_pitch_norm,  # [batch_size, history_length]
            batch_history_roll_ang_vel_norm,  # [batch_size, history_length]
            batch_history_pitch_ang_vel_norm,  # [batch_size, history_length]
            batch_history_roll_trend_norm,  # [batch_size, history_length]
            batch_history_pitch_trend_norm,  # [batch_size, history_length]
            batch_history_time_norm  # [batch_size, history_length, 6] - 注意：这是2D的，需要处理
        )
        
        # 归一化目标数据（只包含roll和pitch，2维）
        # 关键修复：使用 detach() 确保创建的是普通 tensor（不是 inference tensor），可以用于反向传播
        # target 不需要梯度，但必须是普通 tensor 才能用于 loss 计算
        batch_targets_norm = batch_targets.clone().detach()
        batch_targets_norm[:, :, 0] = batch_targets[:, :, 0].clone().detach() * self.input_scale['roll']  # roll
        batch_targets_norm[:, :, 1] = batch_targets[:, :, 1].clone().detach() * self.input_scale['pitch']  # pitch
        # 注意：不再包含角速度归一化，因为输出只包含roll和pitch（2维）
        
        # 前向传播（使用归一化的输入，包含7个特征：roll, pitch, roll_ang_vel, pitch_ang_vel, roll_trend, pitch_trend, time）
        # 关键修复：确保在训练模式下进行前向传播，并检查梯度
        if not self.training:
            # 如果模型不是训练模式，强制设置为训练模式
            self.train()
        
        # 关键修复：确保在启用梯度的上下文中进行前向传播
        # 即使输入数据不需要梯度，只要模型参数需要梯度，输出应该有梯度
        # 使用torch.enable_grad()确保梯度计算被启用
        # 关键：必须在上下文内检查requires_grad，因为退出上下文后可能会丢失信息
        
        # 调试：检查全局梯度状态
        if self.train_step_count % 100 == 0:
            print(f"[梯度调试] 训练前检查:")
            print(f"  - torch.is_grad_enabled(): {torch.is_grad_enabled()}")
            print(f"  - 模型training状态: {self.training}")
            print(f"  - 参数需要梯度: {sum(1 for p in self.parameters() if p.requires_grad)}/{len(list(self.parameters()))}")
            print(f"  - 输入类型: {type(batch_inputs_norm)}, 长度: {len(batch_inputs_norm) if isinstance(batch_inputs_norm, (tuple, list)) else 'N/A'}")
        
        # 关键修复：如果全局梯度被禁用（可能是RL训练循环使用了inference_mode），
        # 我们需要同时使用torch.enable_grad()和torch.inference_mode(False)来确保梯度计算
        # 参考：https://pytorch.org/docs/stable/generated/torch.inference_mode.html
        # inference_mode比no_grad更严格，enable_grad()无法覆盖它，必须使用inference_mode(False)
        with torch.enable_grad():
            with torch.inference_mode(False):
                # 调试：检查上下文内梯度状态
                if self.train_step_count % 100 == 0:
                    print(f"[梯度调试] enable_grad+inference_mode(False)上下文内:")
                    print(f"  - torch.is_grad_enabled(): {torch.is_grad_enabled()}")
                
                # 关键修复：在上下文内重新创建 batch_targets_norm，确保它是普通 tensor（不是 inference tensor）
                # 使用 requires_grad=False 创建普通 tensor，可以用于 loss 计算
                # 注意：只包含roll和pitch（2维）
                batch_targets_norm_in_context = torch.zeros_like(batch_targets, requires_grad=False)
                batch_targets_norm_in_context[:, :, 0] = batch_targets[:, :, 0].clone().detach() * self.input_scale['roll']
                batch_targets_norm_in_context[:, :, 1] = batch_targets[:, :, 1].clone().detach() * self.input_scale['pitch']
                # 注意：不再包含角速度归一化，因为输出只包含roll和pitch（2维）
                
                # 确保所有参数需要梯度
        for param in self.parameters():
            if not param.requires_grad:
                param.requires_grad = True
        
                # 前向传播：将元组解包传递给forward方法
                predicted_states_norm = self.forward(
                    batch_inputs_norm[0],  # history_roll
                    batch_inputs_norm[1],  # history_pitch
                    batch_inputs_norm[2],  # history_roll_ang_vel
                    batch_inputs_norm[3],  # history_pitch_ang_vel
                    batch_inputs_norm[4],  # history_roll_trend
                    batch_inputs_norm[5],  # history_pitch_trend
                    batch_inputs_norm[6]   # history_time
                )  # [batch_size, prediction_steps, 4]
                
                # 关键调试：在上下文内检查预测是否有梯度
                if self.train_step_count % 100 == 0:
                    print(f"[梯度调试] forward后:")
                    print(f"  - predicted_states_norm.requires_grad: {predicted_states_norm.requires_grad}")
                    print(f"  - predicted_states_norm.grad_fn: {predicted_states_norm.grad_fn}")
                    print(f"  - predicted_states_norm.grad_fn类型: {type(predicted_states_norm.grad_fn) if predicted_states_norm.grad_fn is not None else None}")
                
                if not predicted_states_norm.requires_grad:
                    # 如果预测没有梯度，说明计算图被断开了
                    # 关键修复：检查模型参数是否需要梯度
                    params_need_grad = [p.requires_grad for p in self.parameters()]
                    num_params_need_grad = sum(params_need_grad)
                    total_params = len(list(self.parameters()))
                    
                    if self.train_step_count % 100 == 0:
                        print(f"[神经网络训练错误] 预测结果没有梯度！模型training状态: {self.training}, "
                              f"输入requires_grad: {[x.requires_grad if hasattr(x, 'requires_grad') else 'N/A' for x in batch_inputs_norm]}, "
                              f"参数需要梯度: {num_params_need_grad}/{total_params}")
                    
                    # 关键修复：如果参数不需要梯度，强制设置
                    if num_params_need_grad == 0:
                        if self.train_step_count % 100 == 0:
                            print(f"[神经网络训练修复] 所有参数都不需要梯度，强制设置为需要梯度")
                        for param in self.parameters():
                            param.requires_grad = True
                    
                    # 尝试重新计算，确保在训练模式下
                    self.train()
                    # 确保所有子模块也处于训练模式
                    for module in self.modules():
                        if hasattr(module, 'training'):
                            module.training = True
                    # 关键修复：在启用梯度的上下文中重新计算（仍在同一个上下文内）
                    # 同时使用enable_grad和inference_mode(False)确保梯度计算
                    for param in self.parameters():
                        if not param.requires_grad:
                            param.requires_grad = True
                    # 重新计算（仍在同一个上下文内）
                    predicted_states_norm = self.forward(
                        batch_inputs_norm[0],  # history_roll
                        batch_inputs_norm[1],  # history_pitch
                        batch_inputs_norm[2],  # history_roll_ang_vel
                        batch_inputs_norm[3],  # history_pitch_ang_vel
                        batch_inputs_norm[4],  # history_roll_trend
                        batch_inputs_norm[5],  # history_pitch_trend
                        batch_inputs_norm[6]   # history_time
                    )
                    
                    # 再次检查（仍在上下文内）
                    if not predicted_states_norm.requires_grad:
                        if self.train_step_count % 100 == 0:
                            print(f"[神经网络训练严重错误] 重新计算后预测仍然没有梯度！这可能是模型结构问题。")
                        # 详细调试：检查forward过程中的中间结果
                        print(f"[梯度调试] 检查forward内部:")
                        # 手动执行forward的每一步，检查哪一步断开了梯度
                        try:
                            # 测试LSTM：从batch_inputs_norm元组中取第一个样本
                            test_roll = batch_inputs_norm[0][:1]  # [1, history_length]
                            test_pitch = batch_inputs_norm[1][:1]
                            test_roll_vel = batch_inputs_norm[2][:1]
                            test_pitch_vel = batch_inputs_norm[3][:1]
                            test_roll_trend = batch_inputs_norm[4][:1]
                            test_pitch_trend = batch_inputs_norm[5][:1]
                            test_time = batch_inputs_norm[6][:1]  # [1, history_length, 6]
                            
                            # 组合输入
                            test_inputs_stack = torch.cat([
                                test_roll.unsqueeze(-1),  # [1, history_length, 1]
                                test_pitch.unsqueeze(-1),
                                test_roll_vel.unsqueeze(-1),
                                test_pitch_vel.unsqueeze(-1),
                                test_roll_trend.unsqueeze(-1),
                                test_pitch_trend.unsqueeze(-1),
                                test_time  # [1, history_length, 6]
                            ], dim=-1)  # [1, history_length, 12]
                            
                            test_inputs_stack = test_inputs_stack.to(next(self.parameters()).device)
                            lstm_out_test, _ = self.lstm(test_inputs_stack)
                            print(f"  - LSTM输出requires_grad: {lstm_out_test.requires_grad}")
                            print(f"  - LSTM输出grad_fn: {lstm_out_test.grad_fn}")
                        except Exception as e:
                            print(f"  - 调试过程中出错: {e}")
                
                # 关键修复：loss计算必须在同一个上下文内，否则会丢失梯度
                # 计算损失（使用Huber Loss + MSE Loss的组合）
                # Huber Loss在误差较小时表现更好，MSE Loss在误差较大时提供更强的梯度
                mse_loss = self.criterion_mse(predicted_states_norm, batch_targets_norm_in_context)  # [batch_size, prediction_steps, 4]
                huber_loss = self.criterion_huber(predicted_states_norm, batch_targets_norm_in_context)  # [batch_size, prediction_steps, 4]
                
                # 根据误差大小选择损失函数：小误差用Huber Loss，大误差用MSE Loss
                error_magnitude = torch.abs(predicted_states_norm - batch_targets_norm_in_context)
                use_huber = error_magnitude < self.huber_delta
                elementwise_loss = torch.where(use_huber, huber_loss, mse_loss)
                
                # 创建权重矩阵：[batch_size, prediction_steps, 4]
                # 关键改进：降低权重，避免损失值过大导致梯度爆炸和局部最优
                # 权重分配（只包含roll和pitch，2维）：
                # - 姿态（roll, pitch）：权重2.0（降低权重，从3.0降到2.0，避免过度关注）
                # - 第一步预测：权重2.0（降低权重，从3.0降到2.0）
                # - 后续步数：权重1.0
                weights = torch.ones_like(elementwise_loss)
                weights[:, :, 0] = 2.0  # roll权重（从3.0降低到2.0）
                weights[:, :, 1] = 2.0  # pitch权重（从3.0降低到2.0）
                # 注意：不再包含角速度权重，因为输出只包含roll和pitch（2维）
                weights[:, 0, :] = weights[:, 0, :] * 2.0  # 第一步预测权重（从3.0降低到2.0）
                
                # 加权平均损失
                loss = (elementwise_loss * weights).mean()
                
                # 计算每个样本的误差（用于困难样本挖掘）
                sample_errors = torch.sqrt(
                    (predicted_states_norm[:, 0, 0] - batch_targets_norm_in_context[:, 0, 0])**2 +
                    (predicted_states_norm[:, 0, 1] - batch_targets_norm_in_context[:, 0, 1])**2 + 1e-8
                ).detach()  # [batch_size]
                
                # 调试：检查loss是否有梯度（仍在上下文内）
                if self.train_step_count % 100 == 0:
                    print(f"[梯度调试] loss计算后:")
                    print(f"  - loss.requires_grad: {loss.requires_grad}")
                    print(f"  - loss.grad_fn: {loss.grad_fn}")
                    print(f"  - loss值: {loss.item():.6f}")
        
        # 检查loss是否有梯度
        if not loss.requires_grad:
            # 如果loss没有梯度，检查网络参数
            has_grad = any(p.requires_grad for p in self.parameters())
            if not has_grad:
                # 如果所有参数都不需要梯度，设置它们需要梯度
                for param in self.parameters():
                    param.requires_grad = True
                # 重新计算loss（使用归一化的输入）
                # 关键修复：在启用梯度的上下文中重新计算
                with torch.enable_grad():
                    for param in self.parameters():
                        if not param.requires_grad:
                            param.requires_grad = True
                    predicted_states_norm = self.forward(
                        batch_inputs_norm[0],  # history_roll
                        batch_inputs_norm[1],  # history_pitch
                        batch_inputs_norm[2],  # history_roll_ang_vel
                        batch_inputs_norm[3],  # history_pitch_ang_vel
                        batch_inputs_norm[4],  # history_roll_trend
                        batch_inputs_norm[5],  # history_pitch_trend
                        batch_inputs_norm[6]   # history_time
                    )
                # 关键修复：使用criterion_mse而不是criterion（criterion未定义）
                mse_loss = self.criterion_mse(predicted_states_norm, batch_targets_norm)
                huber_loss = self.criterion_huber(predicted_states_norm, batch_targets_norm)
                error_magnitude = torch.abs(predicted_states_norm - batch_targets_norm)
                use_huber = error_magnitude < self.huber_delta
                elementwise_loss = torch.where(use_huber, huber_loss, mse_loss)
                loss = (elementwise_loss * weights).mean()
            else:
                # 如果参数需要梯度但loss没有，可能是计算图被断开
                # 关键修复：检查预测是否有梯度，如果没有，重新计算
                if not predicted_states_norm.requires_grad:
                    # 预测没有梯度，说明计算图被断开，重新计算
                    if self.train_step_count % 100 == 0:
                        print(f"[神经网络训练修复] 预测没有梯度，重新计算。模型training状态: {self.training}")
                    self.train()
                    # 确保所有子模块也处于训练模式
                    for module in self.modules():
                        if hasattr(module, 'training'):
                            module.training = True
                    # 重新计算预测
                    # 关键修复：在启用梯度的上下文中重新计算
                    # 同时使用enable_grad和inference_mode(False)确保梯度计算
                    with torch.enable_grad():
                        with torch.inference_mode(False):
                            for param in self.parameters():
                                if not param.requires_grad:
                                    param.requires_grad = True
                            predicted_states_norm = self.forward(
                                batch_inputs_norm[0],  # history_roll
                                batch_inputs_norm[1],  # history_pitch
                                batch_inputs_norm[2],  # history_roll_ang_vel
                                batch_inputs_norm[3],  # history_pitch_ang_vel
                                batch_inputs_norm[4],  # history_roll_trend
                                batch_inputs_norm[5],  # history_pitch_trend
                                batch_inputs_norm[6]   # history_time
                            )
                    mse_loss = self.criterion_mse(predicted_states_norm, batch_targets_norm)
                    huber_loss = self.criterion_huber(predicted_states_norm, batch_targets_norm)
                    error_magnitude = torch.abs(predicted_states_norm - batch_targets_norm)
                    use_huber = error_magnitude < self.huber_delta
                    elementwise_loss = torch.where(use_huber, huber_loss, mse_loss)
                    loss = (elementwise_loss * weights).mean()
                    # 再次检查loss是否有梯度
                    if not loss.requires_grad:
                        if self.train_step_count % 100 == 0:
                            print(f"[神经网络训练错误] 重新计算后loss仍然没有梯度！模型training状态: {self.training}")
                        return
                else:
                    # 预测有梯度但loss没有，这是异常情况
                    if self.train_step_count % 100 == 0:
                        print(f"[神经网络训练警告] Loss没有梯度但预测有梯度，跳过训练。Loss值: {loss.item():.6f}, "
                              f"模型training状态: {self.training}, 预测requires_grad: {predicted_states_norm.requires_grad}")
                return
        
        # 反向传播
        loss.backward()
        
        # 检查梯度是否正常（如果梯度太小，可能学习太慢）
        # 先计算梯度范数（在裁剪之前）
        total_grad_norm = 0.0
        num_params_with_grad = 0
        for param in self.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_grad_norm += param_norm.item() ** 2
                num_params_with_grad += 1
        total_grad_norm = total_grad_norm ** (1. / 2)
        
        # 关键调试：检查是否有梯度
        if num_params_with_grad == 0:
            # 没有任何参数有梯度，这是严重问题
            if self.train_step_count % 100 == 0:
                print(f"[神经网络训练错误] 没有任何参数有梯度！Loss: {loss.item():.6f}, Loss.requires_grad: {loss.requires_grad}")
            return
        
        if total_grad_norm < 1e-8:  # 降低阈值，允许更小的梯度（从1e-6降低到1e-8）
            # 梯度太小，可能学习太慢，跳过这次更新
            # 但打印警告，帮助调试（使用train_step_count而不是candidate_train_steps，因为此时candidate_train_steps还是0）
            if self.train_step_count % 100 == 0:
                print(f"[神经网络训练警告] 梯度范数太小 ({total_grad_norm:.2e})，跳过更新。Loss: {loss.item():.6f}, 有梯度的参数数: {num_params_with_grad}")
            return
        
        # 关键修复：检测梯度爆炸，如果梯度太大就跳过更新
        max_grad_norm_threshold = 100.0  # 最大允许的梯度范数（裁剪前）
        if total_grad_norm > max_grad_norm_threshold:
            if self.train_step_count % 50 == 0:
                print(f"[神经网络训练警告] 梯度爆炸！梯度范数: {total_grad_norm:.2e} > {max_grad_norm_threshold:.2e}，跳过更新。Loss: {loss.item():.6f}")
            # 清空梯度，避免累积
            self.optimizer.zero_grad()
            return
        
        # 梯度裁剪（防止梯度爆炸，提高训练稳定性）
        # 关键修复：降低梯度裁剪阈值，从1.0降到0.5，更严格地防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        
        # 更新参数（只更新候选模型）
        self.optimizer.step()
        self.optimizer.zero_grad()  # 关键修复：清空梯度，为下次训练做准备
        self.candidate_train_steps += 1
        
        # 关键调试：每次训练都打印（前几次），确认训练真的执行了
        if self.candidate_train_steps <= 5:
            print(f"[神经网络训练成功] 训练步数: {self.candidate_train_steps}, 损失: {loss.item():.6f}, 梯度范数: {total_grad_norm:.6f}")
        
        # 关键调试：确保神经网络真的在训练，打印训练信息
        if self.candidate_train_steps % 50 == 0:  # 每50步打印一次
            first_step_error = torch.sqrt(
                (predicted_states_norm[:, 0, 0] - batch_targets_norm[:, 0, 0])**2 +
                (predicted_states_norm[:, 0, 1] - batch_targets_norm[:, 0, 1])**2
            ).mean().item()
            print(f"[神经网络训练] 训练步数: {self.candidate_train_steps}, 损失: {loss.item():.6f}, "
                  f"第一步预测误差: {first_step_error:.6f} rad, 缓冲区大小: {len(self.training_buffer['inputs'])}, "
                  f"梯度范数: {total_grad_norm:.6f}")
        
        # 更新学习率（使用Warmup + 余弦退火重启 + 自适应学习率 + 更激进的策略）
        if not self._scheduler_initialized:
            # 初始化学习率调度器 - 更激进的策略，帮助跳出局部最优
            self.restart_interval = 300  # 进一步降低重启周期，更频繁地重启
            self.warmup_steps = 50  # 减少warmup步数，更快进入训练
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.restart_interval,  # 初始周期长度
                T_mult=1,  # 周期倍增因子（1表示周期长度不变）
                eta_min=self.learning_rate * 0.001,  # 最小学习率（更小，帮助跳出局部最优）
            )
            self._scheduler_initialized = True
            # 记录损失历史，用于检测是否陷入局部最优
            self.loss_history = []
            self.loss_history_window = 100  # 记录最近100步的损失
        
        # Warmup阶段：前warmup_steps步线性增加学习率
        if self.candidate_train_steps <= self.warmup_steps:
            warmup_lr = self.learning_rate * (self.candidate_train_steps / self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr
        else:
            # 使用余弦退火重启
            if self._scheduler_initialized:
                self.scheduler.step()
            
            # 关键改进：检测局部最优并主动跳出
            # 记录损失历史
            if not hasattr(self, 'loss_history'):
                self.loss_history = []
                self.loss_history_window = 100
            if len(self.loss_history) >= self.loss_history_window:
                self.loss_history.pop(0)
            self.loss_history.append(loss.item())
            
            # 如果损失长时间不下降，主动调整学习率
            if self.candidate_train_steps % 100 == 0 and self.candidate_train_steps > 200 and len(self.loss_history) >= 50:
                recent_losses = self.loss_history[-50:]
                early_losses = self.loss_history[-100:-50] if len(self.loss_history) >= 100 else self.loss_history[:50]
                
                # 计算损失变化率
                if len(early_losses) > 0:
                    early_avg = sum(early_losses) / len(early_losses)
                    recent_avg = sum(recent_losses) / len(recent_losses)
                    loss_improvement = (early_avg - recent_avg) / max(early_avg, 1e-8)
                    
                    current_lr = self.optimizer.param_groups[0]['lr']
                    
                    # 如果损失改善很小（<1%），认为可能陷入局部最优
                    if loss_improvement < 0.01 and current_lr < self.learning_rate * 0.5:
                        # 增加学习率，帮助跳出局部最优
                        new_lr = min(current_lr * 1.5, self.learning_rate * 0.8)  # 最多恢复到初始学习率的80%
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        if self.candidate_train_steps % 500 == 0:
                            print(f"[神经网络训练] 检测到局部最优，主动增加学习率：{current_lr:.6f} -> {new_lr:.6f} (损失改善率: {loss_improvement*100:.2f}%)")
                    # 如果损失改善很好（>5%），可以稍微降低学习率，更精细地优化
                    elif loss_improvement > 0.05 and current_lr > self.learning_rate * 0.1:
                        new_lr = max(current_lr * 0.9, self.learning_rate * 0.1)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
        
        # 更新困难样本缓冲区（保留预测误差较大的样本）
        if len(sample_errors) > 0:
            # 找到误差较大的样本
            error_threshold = sample_errors.quantile(0.7).item()  # 前30%的困难样本
            hard_mask = sample_errors > error_threshold
            
            for i, is_hard in enumerate(hard_mask):
                if is_hard:
                    # 添加到困难样本缓冲区
                    self.hard_sample_buffer['inputs'].append(batch_inputs_list[i])
                    self.hard_sample_buffer['targets'].append(batch_targets_list[i])
                    self.hard_sample_buffer['errors'].append(sample_errors[i].item())
            
            # 限制困难样本缓冲区大小（保留误差最大的样本）
            if len(self.hard_sample_buffer['inputs']) > self.hard_sample_buffer['max_size']:
                # 按误差排序，保留误差最大的样本
                sorted_indices = sorted(
                    range(len(self.hard_sample_buffer['errors'])),
                    key=lambda i: self.hard_sample_buffer['errors'][i],
                    reverse=True
                )[:self.hard_sample_buffer['max_size']]
                
                self.hard_sample_buffer['inputs'] = [self.hard_sample_buffer['inputs'][i] for i in sorted_indices]
                self.hard_sample_buffer['targets'] = [self.hard_sample_buffer['targets'][i] for i in sorted_indices]
                self.hard_sample_buffer['errors'] = [self.hard_sample_buffer['errors'][i] for i in sorted_indices]
        
        # 定期打印训练信息（每100次训练打印一次）
        if self.candidate_train_steps % 100 == 0:
            # 计算反归一化后的预测误差（用于监控，只包含roll和pitch，2维）
            predicted_states_denorm = predicted_states_norm.clone()
            predicted_states_denorm[:, :, 0] = predicted_states_norm[:, :, 0] / self.input_scale['roll']
            predicted_states_denorm[:, :, 1] = predicted_states_norm[:, :, 1] / self.input_scale['pitch']
            # 注意：不再包含角速度反归一化，因为输出只包含roll和pitch（2维）
            
            # 计算第一步预测的误差（奖励函数主要使用第一步）
            first_step_error = torch.sqrt(
                (predicted_states_denorm[:, 0, 0] - batch_targets[:, 0, 0])**2 +
                (predicted_states_denorm[:, 0, 1] - batch_targets[:, 0, 1])**2 + 1e-8
            ).mean().item()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            # 计算预测值的范围（用于确认预测是否在变化）
            pred_roll_range = (predicted_states_denorm[:, 0, 0].max() - predicted_states_denorm[:, 0, 0].min()).item()
            pred_pitch_range = (predicted_states_denorm[:, 0, 1].max() - predicted_states_denorm[:, 0, 1].min()).item()
            
            print(f"[神经网络训练] 训练步数: {self.candidate_train_steps}, 归一化损失: {loss.item():.6f}, "
                  f"第一步预测误差: {first_step_error:.6f} rad, 缓冲区大小: {len(self.training_buffer['inputs'])}, "
                  f"梯度范数: {total_grad_norm:.6f}, 学习率: {current_lr:.8f}, "
                  f"预测roll范围: {pred_roll_range:.6f}, 预测pitch范围: {pred_pitch_range:.6f}")
        
        # 清空缓冲区（可选：保留一些样本用于下次训练）
        # 这里我们保留最后一半的样本
        keep_size = len(self.training_buffer['inputs']) // 2
        if keep_size > 0:
            self.training_buffer['inputs'] = self.training_buffer['inputs'][-keep_size:]
            self.training_buffer['targets'] = self.training_buffer['targets'][-keep_size:]
    
    def evaluate_prediction_quality_from_observations(
        self,
        observation_history: dict,  # 机器狗观测历史
        platform_history: dict,  # 平台历史数据（用于计算真实目标）
        delay_steps: int = 5
    ) -> bool:
        """使用机器狗观测历史评估预测质量
        
        评估方法：从历史数据中取多个时间点，对于每个时间点 t，
        使用 t-delay_steps 时刻的机器狗观测历史作为输入，
        预测未来多步的平台状态，然后与实际的平台状态比较。
        如果大部分预测都很准确，就认为网络训练好了。
        
        Args:
            observation_history: 机器狗观测历史字典，包含：
                - 'base_lin_vel': 历史列表 [num_envs, 3]
                - 'base_ang_vel': 历史列表 [num_envs, 3]
                - 'projected_gravity': 历史列表 [num_envs, 3]
                - 'velocity_commands': 历史列表 [num_envs, num_velocity_commands]
                - 'joint_pos': 历史列表 [num_envs, num_joints]
                - 'joint_vel': 历史列表 [num_envs, num_joints]
                - 'actions': 历史列表 [num_envs, num_actions]
            platform_history: 平台历史数据字典，包含：
                - 'quat_w': 历史四元数列表 [num_envs, 4]
                - 'ang_vel_w': 历史角速度列表 [num_envs, 3]
            delay_steps: 延迟步数，与训练时保持一致（默认5）
        
        Returns:
            True: 预测质量足够好，可以使用神经网络预测
            False: 预测质量不够好，应该使用线性外推
        """
        from isaaclab.utils.math import euler_xyz_from_quat
        
        # 检查数据是否足够
        if platform_history.get('quat_w') is None or platform_history.get('ang_vel_w') is None:
            return False
        
        # 获取观测历史长度（所有观测项应该有相同的长度）
        obs_history_length = len(observation_history.get('base_lin_vel', []))
        if obs_history_length < self.history_length:
            print(f"[预测质量评估] 无法评估: 观测历史数据不足 ({obs_history_length} < {self.history_length})")
            return False
        
        # 获取平台历史长度
        quat_history = platform_history['quat_w']
        ang_vel_history = platform_history['ang_vel_w']
        
        # 处理列表或tensor格式
        if isinstance(quat_history, list):
            quat_history = torch.stack(quat_history, dim=0)  # [history_length, num_envs, 4]
        if isinstance(ang_vel_history, list):
            ang_vel_history = torch.stack(ang_vel_history, dim=0)  # [history_length, num_envs, 3]
        
        platform_history_length = quat_history.shape[0]
        
        # 只使用环境0的数据进行评估（因为所有环境的平台运动是一致的）
        env_idx = 0
        
        # 需要足够的历史数据：至少需要 history_length + prediction_steps 个历史点
        min_required_length = self.history_length + self.prediction_steps
        if platform_history_length < min_required_length:
            print(f"[预测质量评估] 无法评估: 平台历史数据不足 ({platform_history_length} < {min_required_length})")
            return False
        
        # 评估多个时间点
        evaluation_errors = []
        
        # 选择评估的时间点
        # 从 history_length + delay_steps 开始（需要 history_length 个观测历史 + delay_steps 个延迟）
        eval_start_idx = self.history_length + delay_steps
        eval_end_idx = min(obs_history_length, platform_history_length) - self.prediction_steps
        
        if eval_end_idx < eval_start_idx:
            print(f"[预测质量评估] 无法评估: 评估范围不足 (start={eval_start_idx}, end={eval_end_idx})")
            return False
        
        # 计算评估步长：尽量评估 min_evaluation_samples 个点
        available_points = eval_end_idx - eval_start_idx + 1
        num_evaluation_points = min(self.min_evaluation_samples, available_points)
        eval_step = max(1, available_points // num_evaluation_points) if num_evaluation_points > 0 else 1
        
        for t_idx in range(eval_start_idx, eval_end_idx + 1, eval_step):
            # 获取 t_idx 时刻之前的机器狗观测历史（history_length 个点）
            # 使用 t_idx - delay_steps 之前的观测历史（模拟延迟观测）
            obs_start_idx = max(0, t_idx - delay_steps - self.history_length)
            obs_end_idx = t_idx - delay_steps
            
            if obs_end_idx <= obs_start_idx:
                continue  # 跳过这个时间点
            
            # 提取机器狗观测历史（只使用环境0）
            obs_base_lin_vel = observation_history['base_lin_vel'][obs_start_idx:obs_end_idx]
            obs_base_ang_vel = observation_history['base_ang_vel'][obs_start_idx:obs_end_idx]
            obs_projected_gravity = observation_history['projected_gravity'][obs_start_idx:obs_end_idx]
            obs_velocity_commands = observation_history['velocity_commands'][obs_start_idx:obs_end_idx]
            obs_joint_pos = observation_history['joint_pos'][obs_start_idx:obs_end_idx]
            obs_joint_vel = observation_history['joint_vel'][obs_start_idx:obs_end_idx]
            obs_actions = observation_history['actions'][obs_start_idx:obs_end_idx]
            
            # 堆叠为tensor（只取环境0）
            actual_obs_length = len(obs_base_lin_vel)
            if actual_obs_length < self.history_length:
                # 如果历史数据不足，用第一个值填充
                padding_size = self.history_length - actual_obs_length
                obs_base_lin_vel = [obs_base_lin_vel[0]] * padding_size + obs_base_lin_vel
                obs_base_ang_vel = [obs_base_ang_vel[0]] * padding_size + obs_base_ang_vel
                obs_projected_gravity = [obs_projected_gravity[0]] * padding_size + obs_projected_gravity
                obs_velocity_commands = [obs_velocity_commands[0]] * padding_size + obs_velocity_commands
                obs_joint_pos = [obs_joint_pos[0]] * padding_size + obs_joint_pos
                obs_joint_vel = [obs_joint_vel[0]] * padding_size + obs_joint_vel
                obs_actions = [obs_actions[0]] * padding_size + obs_actions
            
            # 只取最后 history_length 个点
            obs_base_lin_vel = obs_base_lin_vel[-self.history_length:]
            obs_base_ang_vel = obs_base_ang_vel[-self.history_length:]
            obs_projected_gravity = obs_projected_gravity[-self.history_length:]
            obs_velocity_commands = obs_velocity_commands[-self.history_length:]
            obs_joint_pos = obs_joint_pos[-self.history_length:]
            obs_joint_vel = obs_joint_vel[-self.history_length:]
            obs_actions = obs_actions[-self.history_length:]
            
            # 堆叠为tensor（只取环境0）
            obs_history_tensor = {
                'base_lin_vel': torch.stack(obs_base_lin_vel, dim=1)[0:1],  # [1, history_length, 3]
                'base_ang_vel': torch.stack(obs_base_ang_vel, dim=1)[0:1],  # [1, history_length, 3]
                'projected_gravity': torch.stack(obs_projected_gravity, dim=1)[0:1],  # [1, history_length, 3]
                'velocity_commands': torch.stack(obs_velocity_commands, dim=1)[0:1],  # [1, history_length, num_velocity_commands]
                'joint_pos': torch.stack(obs_joint_pos, dim=1)[0:1],  # [1, history_length, num_joints]
                'joint_vel': torch.stack(obs_joint_vel, dim=1)[0:1],  # [1, history_length, num_joints]
                'actions': torch.stack(obs_actions, dim=1)[0:1],  # [1, history_length, num_actions]
            }
            
            # 关键修复：使用与训练时相同的归一化方式
            device = obs_history_tensor['base_lin_vel'].device
            obs_norm = {}
            if self.obs_stats['base_lin_vel']['mean'] is not None:
                # 使用保存的统计信息进行归一化
                for key in obs_history_tensor.keys():
                    if key in self.obs_stats and self.obs_stats[key]['mean'] is not None:
                        mean = self.obs_stats[key]['mean'].to(device)
                        std = self.obs_stats[key]['std'].to(device)
                        # 扩展维度以匹配输入形状
                        while mean.dim() < obs_history_tensor[key].dim():
                            mean = mean.unsqueeze(0)
                            std = std.unsqueeze(0)
                        # 归一化到[-1, 1]
                        obs_norm[key] = (obs_history_tensor[key] - mean) / (std * 3.0 + 1e-8)
                        obs_norm[key] = torch.clamp(obs_norm[key], -1.0, 1.0)
                    else:
                        obs_norm[key] = obs_history_tensor[key]
            else:
                # 如果还没有统计信息，使用原始数据（可能模型还没训练）
                obs_norm = obs_history_tensor
            
            # 预测未来多步状态（使用生产模型进行评估）
            if self.production_model is None:
                self.eval()
                with torch.no_grad():
                    predicted_states_norm = self.forward(
                        obs_norm['base_lin_vel'],
                        obs_norm['base_ang_vel'],
                        obs_norm['projected_gravity'],
                        obs_norm['velocity_commands'],
                        obs_norm['joint_pos'],
                        obs_norm['joint_vel'],
                        obs_norm['actions'],
                    )  # [1, prediction_steps, 2] (归一化的)
            else:
                self.production_model.eval()
                with torch.no_grad():
                    predicted_states_norm = self._forward_with_model(
                        self.production_model,
                        obs_norm['base_lin_vel'],
                        obs_norm['base_ang_vel'],
                        obs_norm['projected_gravity'],
                        obs_norm['velocity_commands'],
                        obs_norm['joint_pos'],
                        obs_norm['joint_vel'],
                        obs_norm['actions'],
                    )  # [1, prediction_steps, 2] (归一化的)
            
            # 关键修复：反归一化输出（从归一化空间转换回原始空间）
            if self.target_stats['mean'] is not None:
                target_mean = self.target_stats['mean'].to(device)  # [4]
                target_std = self.target_stats['std'].to(device)  # [4]
                # 扩展维度以匹配输出形状
                target_mean = target_mean.unsqueeze(0).unsqueeze(0)  # [1, 1, 4]
                target_std = target_std.unsqueeze(0).unsqueeze(0)  # [1, 1, 4]
                # 反归一化
                predicted_states = predicted_states_norm * (target_std * 3.0) + target_mean
            else:
                # 如果没有归一化统计信息，直接使用输出（可能不准确）
                predicted_states = predicted_states_norm
            
            # 反归一化输出（如果使用了归一化）
            # 注意：现在网络输出可能已经是归一化的，需要检查
            # 为了简化，假设输出已经是反归一化的（或者需要根据实际情况调整）
            
            # 评估所有预测步数
            step_errors = []
            for step_idx in range(self.prediction_steps):
                predicted_state = predicted_states[0, step_idx, :]  # [2] (roll, pitch)
                predicted_roll = predicted_state[0]
                predicted_pitch = predicted_state[1]
                # 角速度暂时设为0（如果需要可以后续添加）
                predicted_roll_ang_vel = torch.tensor(0.0, device=predicted_roll.device)
                predicted_pitch_ang_vel = torch.tensor(0.0, device=predicted_roll.device)
                
                # 计算对应时刻的实际平台状态（从 t_idx 开始，未来 step_idx 步）
                target_idx = t_idx + step_idx
                if target_idx < platform_history_length:
                    target_quat = quat_history[target_idx, env_idx:env_idx+1, :]  # [1, 4]
                    target_ang_vel = ang_vel_history[target_idx, env_idx:env_idx+1, :]  # [1, 3]
                    actual_state_roll, actual_state_pitch, _ = euler_xyz_from_quat(target_quat)
                    actual_state_roll_ang_vel = target_ang_vel[0, 0]
                    actual_state_pitch_ang_vel = target_ang_vel[0, 1]
                else:
                    continue  # 跳过这个步数
                
                # 计算预测误差：使用所有4个量（roll, pitch, roll_ang_vel, pitch_ang_vel）
                # 姿态误差（rad）
                orientation_error = torch.sqrt(
                    (predicted_roll - actual_state_roll[0])**2 +
                    (predicted_pitch - actual_state_pitch[0])**2 + 1e-8
                )
                # 角速度误差（rad/s），归一化到rad单位
                angular_velocity_error = torch.sqrt(
                    (predicted_roll_ang_vel - actual_state_roll_ang_vel)**2 +
                    (predicted_pitch_ang_vel - actual_state_pitch_ang_vel)**2 + 1e-8
                )
                # 综合误差：姿态误差（权重0.7）+ 角速度误差（权重0.3，归一化到rad单位）
                angular_velocity_error_normalized = angular_velocity_error / 0.5  # 归一化到rad单位
                combined_error = 0.7 * orientation_error + 0.3 * angular_velocity_error_normalized
                
                step_errors.append(combined_error.item())
            
            # 使用所有步数的平均误差作为该时间点的误差
            if step_errors:
                mean_error = sum(step_errors) / len(step_errors)
                evaluation_errors.append(mean_error)
        
        # 如果没有任何评估样本，认为预测质量不够好
        if len(evaluation_errors) == 0:
            print(f"[预测质量评估] 无法评估: 没有评估样本")
            return False
        
        # 计算有多少比例的预测误差小于阈值
        accurate_predictions = sum(1 for err in evaluation_errors if err < self.prediction_quality_threshold)
        accuracy_ratio = accurate_predictions / len(evaluation_errors)
        
        # 计算误差统计信息
        mean_error = sum(evaluation_errors) / len(evaluation_errors) if evaluation_errors else float('inf')
        max_error = max(evaluation_errors) if evaluation_errors else float('inf')
        min_error = min(evaluation_errors) if evaluation_errors else float('inf')
        median_error = sorted(evaluation_errors)[len(evaluation_errors) // 2] if evaluation_errors else float('inf')
        
        # 如果至少达到要求的准确率，认为网络训练好了
        is_good = accuracy_ratio >= self.evaluation_accuracy_ratio
        
        # 计算样本数说明
        sample_info = f"样本数={len(evaluation_errors)} (从观测历史长度{obs_history_length}和平台历史长度{platform_history_length}中，从索引{eval_start_idx}到{eval_end_idx}，步长{eval_step}，每个样本评估{self.prediction_steps}步)"
        
        # 保存评估信息，供后续查询
        self._last_evaluation_info = {
            'is_good': is_good,
            'accuracy_ratio': accuracy_ratio,
            'required_accuracy_ratio': self.evaluation_accuracy_ratio,
            'accurate_predictions': accurate_predictions,
            'total_samples': len(evaluation_errors),
            'mean_error': mean_error,
            'max_error': max_error,
            'min_error': min_error,
            'median_error': median_error,
            'threshold': self.prediction_quality_threshold,
            'sample_info': sample_info,
        }
        
        if is_good:
            self.prediction_quality_verified = True
            print(f"[预测质量评估] 评估通过: 准确率 {accuracy_ratio:.2%} >= {self.evaluation_accuracy_ratio:.2%}, {sample_info}")
        else:
            print(f"[预测质量评估] 评估未通过: 准确率 {accuracy_ratio:.2%} < {self.evaluation_accuracy_ratio:.2%}, "
                  f"{sample_info}, 阈值={self.prediction_quality_threshold:.4f} rad, "
                  f"平均误差={mean_error:.4f} rad (姿态70%+角速度30%), 最大误差={max_error:.4f} rad, 最小误差={min_error:.4f} rad, "
                  f"中位数误差={median_error:.4f} rad")
        
        return is_good
    
    def evaluate_prediction_quality(self, platform_history: dict, delay_steps: int = 5) -> bool:
        """使用历史数据评估预测质量（旧方法，已废弃，保留用于兼容性）
        
        注意：此方法已废弃，请使用 evaluate_prediction_quality_from_observations
        """
        # 此方法已废弃，但保留用于兼容性
        # 实际应该使用 evaluate_prediction_quality_from_observations
        return False
    
    def is_prediction_quality_good(self) -> bool:
        """检查预测质量是否足够好（已验证）
        
        一旦评估通过，就持续使用网络预测（不再每次都检查）。
        
        Returns:
            True: 预测质量足够好，可以使用神经网络预测
            False: 预测质量不够好，应该使用线性外推
        """
        return self.prediction_quality_verified
    
    def get_prediction_quality_info(self) -> dict:
        """获取预测质量的详细信息
        
        Returns:
            包含预测质量信息的字典：
            - 'is_good': bool, 预测质量是否足够好
            - 'verified': bool, 是否已验证
            - 'last_evaluation_info': dict, 最后一次评估的详细信息（如果存在）
        """
        info = {
            'is_good': self.prediction_quality_verified,
            'verified': self.prediction_quality_verified,
        }
        
        # 如果有最后一次评估的信息，添加到返回字典中
        if hasattr(self, '_last_evaluation_info'):
            info['last_evaluation_info'] = self._last_evaluation_info
        else:
            info['last_evaluation_info'] = None
        
        return info
    
    def evaluate_and_update_production_model(self, platform_history: dict, delay_steps: int = 5, 
                                             current_step: int = 0, observation_history: dict = None) -> bool:
        """评估候选模型，如果更好则替换生产模型
        
        Args:
            platform_history: 平台历史数据字典
            delay_steps: 延迟步数
            current_step: 当前步数（用于判断是否需要评估）
            observation_history: 机器狗观测历史（可选，如果提供则使用，否则无法评估）
        
        Returns:
            True: 候选模型更好，已替换生产模型
            False: 候选模型不够好，未替换生产模型
        """
        # 检查是否需要评估（每隔一定步数评估一次）
        if current_step - self.last_evaluation_step < self.evaluation_interval:
            return False
        
        # 检查候选模型是否已经训练过
        if self.candidate_train_steps < 10:  # 至少训练10步才评估
            return False
        
        # 关键修复：如果没有提供观测历史，无法评估（因为模型现在需要观测历史作为输入）
        if observation_history is None:
            return False
        
        # 评估候选模型和生产模型的性能（使用观测历史）
        candidate_score = self._evaluate_model_performance(
            self, platform_history, delay_steps, observation_history=observation_history, model_name="候选模型"
        )
        
        # 如果生产模型未初始化，使用候选模型的分数（初始时两个模型相同）
        if self.production_model is None:
            production_score = candidate_score
        else:
            production_score = self._evaluate_model_performance(
                self.production_model, platform_history, delay_steps, observation_history=observation_history, model_name="生产模型"
            )
        
        # 如果候选模型更好（误差更小），则替换生产模型
        if candidate_score < production_score:
            # 替换生产模型（手动复制参数，避免deepcopy的递归问题）
            with torch.no_grad():
                # 复制LSTM参数
                for prod_param, cand_param in zip(self.production_model['lstm'].parameters(), self.lstm.parameters()):
                    prod_param.data.copy_(cand_param.data)
                # 复制FC层参数（ResNet风格）
                for prod_param, cand_param in zip(self.production_model['fc1'].parameters(), self.fc1.parameters()):
                    prod_param.data.copy_(cand_param.data)
                for prod_param, cand_param in zip(self.production_model['ln1'].parameters(), self.ln1.parameters()):
                    prod_param.data.copy_(cand_param.data)
                for prod_param, cand_param in zip(self.production_model['attention'].parameters(), self.attention.parameters()):
                    prod_param.data.copy_(cand_param.data)
                for prod_param, cand_param in zip(self.production_model['attention_norm'].parameters(), self.attention_norm.parameters()):
                    prod_param.data.copy_(cand_param.data)
                for prod_param, cand_param in zip(self.production_model['res_block1'].parameters(), self.res_block1.parameters()):
                    prod_param.data.copy_(cand_param.data)
                for prod_param, cand_param in zip(self.production_model['res_block2'].parameters(), self.res_block2.parameters()):
                    prod_param.data.copy_(cand_param.data)
                # 注意：模型只有res_block1和res_block2，没有res_block3
                for prod_param, cand_param in zip(self.production_model['fc_out'].parameters(), self.fc_out.parameters()):
                    prod_param.data.copy_(cand_param.data)
            
            # 生产模型不需要梯度
            for param in self.production_model.parameters():
                param.requires_grad = False
            self.production_model.eval()
            
            self.last_evaluation_step = current_step
            
            # ========== 打印参数更新提示 ==========
            improvement_ratio = (production_score - candidate_score) / production_score * 100  # 改进百分比
            print("=" * 80)
            print(f"[神经网络参数更新] ⚡ 候选模型性能更好，已更新生产模型参数！")
            print(f"  - 当前步数: {current_step}")
            print(f"  - 候选模型训练步数: {self.candidate_train_steps}")
            print(f"  - 候选模型误差: {candidate_score:.6f} rad")
            print(f"  - 原生产模型误差: {production_score:.6f} rad")
            print(f"  - 性能提升: {improvement_ratio:.2f}% (误差降低 {production_score - candidate_score:.6f} rad)")
            print(f"  - 评估间隔: {self.evaluation_interval} 步")
            print("=" * 80)
            
            return True
        else:
            self.last_evaluation_step = current_step
            # 只在调试时打印（避免输出过多）
            if current_step % (self.evaluation_interval * 5) == 0:  # 每5次评估打印一次
                print(f"[神经网络评估] 候选模型不够好（误差: {candidate_score:.6f} >= {production_score:.6f}），保持生产模型")
            return False
    
    def _evaluate_model_performance(self, model, platform_history: dict, delay_steps: int = 5, 
                                    observation_history: dict = None, model_name: str = "模型") -> float:
        """评估模型性能（返回平均预测误差）
        
        Args:
            model: 要评估的模型
            platform_history: 平台历史数据字典
            delay_steps: 延迟步数
            observation_history: 机器狗观测历史（可选，如果提供则使用，否则使用平台历史）
            model_name: 模型名称（用于打印）
        
        Returns:
            平均预测误差（越小越好）
        """
        from isaaclab.utils.math import euler_xyz_from_quat
        
        # 关键修复：如果提供了观测历史，使用观测历史进行评估（与训练时一致）
        if observation_history is not None:
            # 使用观测历史进行评估（与训练时一致）
            return self._evaluate_model_performance_from_observations(
                model, observation_history, platform_history, delay_steps, model_name
            )
        
        # 如果没有提供观测历史，使用平台历史（旧方法，已废弃但保留兼容性）
        if platform_history.get('quat_w') is None or platform_history.get('ang_vel_w') is None:
            return float('inf')
        
        # 旧方法：直接使用平台历史进行评估（已废弃，但保留兼容性）
        # 注意：这种方法不准确，因为模型现在需要机器狗观测历史作为输入
        return float('inf')  # 如果没有观测历史，返回无穷大，表示无法评估
    
    def _evaluate_model_performance_from_observations(
        self, model, observation_history: dict, platform_history: dict, 
        delay_steps: int = 5, model_name: str = "模型"
    ) -> float:
        """使用机器狗观测历史评估模型性能（新方法，与训练时一致）
        
        Args:
            model: 要评估的模型
            observation_history: 机器狗观测历史字典
            platform_history: 平台历史数据字典（用于计算真实目标）
            delay_steps: 延迟步数
            model_name: 模型名称（用于打印）
        
        Returns:
            平均预测误差（越小越好）
        """
        from isaaclab.utils.math import euler_xyz_from_quat
        
        if platform_history.get('quat_w') is None or platform_history.get('ang_vel_w') is None:
            return float('inf')
        
        # 获取观测历史长度
        obs_history_length = len(observation_history.get('base_lin_vel', []))
        if obs_history_length < self.history_length:
            return float('inf')
        
        quat_history = platform_history['quat_w']
        ang_vel_history = platform_history['ang_vel_w']
        
        # 处理列表或tensor格式
        if isinstance(quat_history, list):
            quat_history = torch.stack(quat_history, dim=0)
        if isinstance(ang_vel_history, list):
            ang_vel_history = torch.stack(ang_vel_history, dim=0)
        
        platform_history_length = quat_history.shape[0]
        num_envs = quat_history.shape[1]
        
        # 需要足够的历史数据
        min_required_length = delay_steps + self.history_length + self.prediction_steps
        if platform_history_length < min_required_length or obs_history_length < min_required_length:
            return float('inf')
        
        # 评估多个时间点（使用较少的样本，加快评估速度）
        evaluation_errors = []
        eval_start_idx = delay_steps + self.history_length
        eval_end_idx = min(obs_history_length, platform_history_length) - self.prediction_steps
        eval_samples = min(20, (eval_end_idx - eval_start_idx) // 2)  # 最多评估20个样本
        eval_step = max(1, (eval_end_idx - eval_start_idx) // eval_samples) if eval_samples > 0 else 1
        
        model.eval()
        with torch.no_grad():
            for t_idx in range(eval_start_idx, eval_end_idx, eval_step):
                # 获取实际 t 时刻的状态
                if t_idx >= platform_history_length:
                    continue
                actual_quat = quat_history[t_idx]
                actual_ang_vel = ang_vel_history[t_idx]
                
                actual_roll, actual_pitch, _ = euler_xyz_from_quat(actual_quat)
                
                # 使用 t-delay_steps 之前的机器狗观测历史（与训练时一致）
                obs_start_idx = max(0, t_idx - delay_steps - self.history_length)
                obs_end_idx = t_idx - delay_steps
                
                if obs_end_idx <= obs_start_idx or obs_end_idx > obs_history_length:
                    continue
                
                # 提取机器狗观测历史（只使用环境0）
                obs_base_lin_vel = observation_history['base_lin_vel'][obs_start_idx:obs_end_idx]
                obs_base_ang_vel = observation_history['base_ang_vel'][obs_start_idx:obs_end_idx]
                obs_projected_gravity = observation_history['projected_gravity'][obs_start_idx:obs_end_idx]
                obs_velocity_commands = observation_history['velocity_commands'][obs_start_idx:obs_end_idx]
                obs_joint_pos = observation_history['joint_pos'][obs_start_idx:obs_end_idx]
                obs_joint_vel = observation_history['joint_vel'][obs_start_idx:obs_end_idx]
                obs_actions = observation_history['actions'][obs_start_idx:obs_end_idx]
                
                # 确保有足够的历史数据
                actual_obs_length = len(obs_base_lin_vel)
                if actual_obs_length < self.history_length:
                    # 如果历史数据不足，用第一个值填充
                    padding_size = self.history_length - actual_obs_length
                    obs_base_lin_vel = [obs_base_lin_vel[0]] * padding_size + obs_base_lin_vel
                    obs_base_ang_vel = [obs_base_ang_vel[0]] * padding_size + obs_base_ang_vel
                    obs_projected_gravity = [obs_projected_gravity[0]] * padding_size + obs_projected_gravity
                    obs_velocity_commands = [obs_velocity_commands[0]] * padding_size + obs_velocity_commands
                    obs_joint_pos = [obs_joint_pos[0]] * padding_size + obs_joint_pos
                    obs_joint_vel = [obs_joint_vel[0]] * padding_size + obs_joint_vel
                    obs_actions = [obs_actions[0]] * padding_size + obs_actions
                
                # 只取最后 history_length 个点
                obs_base_lin_vel = obs_base_lin_vel[-self.history_length:]
                obs_base_ang_vel = obs_base_ang_vel[-self.history_length:]
                obs_projected_gravity = obs_projected_gravity[-self.history_length:]
                obs_velocity_commands = obs_velocity_commands[-self.history_length:]
                obs_joint_pos = obs_joint_pos[-self.history_length:]
                obs_joint_vel = obs_joint_vel[-self.history_length:]
                obs_actions = obs_actions[-self.history_length:]
                
                # 堆叠为tensor（只取环境0）
                obs_history_tensor = {
                    'base_lin_vel': torch.stack(obs_base_lin_vel, dim=1)[0:1],  # [1, history_length, 3]
                    'base_ang_vel': torch.stack(obs_base_ang_vel, dim=1)[0:1],  # [1, history_length, 3]
                    'projected_gravity': torch.stack(obs_projected_gravity, dim=1)[0:1],  # [1, history_length, 3]
                    'velocity_commands': torch.stack(obs_velocity_commands, dim=1)[0:1],  # [1, history_length, num_velocity_commands]
                    'joint_pos': torch.stack(obs_joint_pos, dim=1)[0:1],  # [1, history_length, num_joints]
                    'joint_vel': torch.stack(obs_joint_vel, dim=1)[0:1],  # [1, history_length, num_joints]
                    'actions': torch.stack(obs_actions, dim=1)[0:1],  # [1, history_length, num_actions]
                }
                
                # 关键修复：使用与训练时相同的归一化方式
                device = obs_history_tensor['base_lin_vel'].device
                obs_norm = {}
                if self.obs_stats['base_lin_vel']['mean'] is not None:
                    # 使用保存的统计信息进行归一化
                    for key in obs_history_tensor.keys():
                        if key in self.obs_stats and self.obs_stats[key]['mean'] is not None:
                            mean = self.obs_stats[key]['mean'].to(device)
                            std = self.obs_stats[key]['std'].to(device)
                            # 扩展维度以匹配输入形状
                            while mean.dim() < obs_history_tensor[key].dim():
                                mean = mean.unsqueeze(0)
                                std = std.unsqueeze(0)
                            # 归一化到[-1, 1]
                            obs_norm[key] = (obs_history_tensor[key] - mean) / (std * 3.0 + 1e-8)
                            obs_norm[key] = torch.clamp(obs_norm[key], -1.0, 1.0)
                        else:
                            obs_norm[key] = obs_history_tensor[key]
                else:
                    # 如果还没有统计信息，使用原始数据
                    obs_norm = obs_history_tensor
                
                # 预测
                if isinstance(model, nn.ModuleDict):
                    predicted_states_norm = self._forward_with_model(
                        model,
                        obs_norm['base_lin_vel'],
                        obs_norm['base_ang_vel'],
                        obs_norm['projected_gravity'],
                        obs_norm['velocity_commands'],
                        obs_norm['joint_pos'],
                        obs_norm['joint_vel'],
                        obs_norm['actions'],
                    )  # [1, prediction_steps, 2] (归一化的)
                else:
                    predicted_states_norm = model.forward(
                        obs_norm['base_lin_vel'],
                        obs_norm['base_ang_vel'],
                        obs_norm['projected_gravity'],
                        obs_norm['velocity_commands'],
                        obs_norm['joint_pos'],
                        obs_norm['joint_vel'],
                        obs_norm['actions'],
                    )  # [1, prediction_steps, 2] (归一化的)
                
                # 关键修复：反归一化输出（从归一化空间转换回原始空间）
                if self.target_stats['mean'] is not None:
                    target_mean = self.target_stats['mean'].to(device)  # [4]
                    target_std = self.target_stats['std'].to(device)  # [4]
                    # 扩展维度以匹配输出形状
                    target_mean = target_mean.unsqueeze(0).unsqueeze(0)  # [1, 1, 4]
                    target_std = target_std.unsqueeze(0).unsqueeze(0)  # [1, 1, 4]
                    # 反归一化
                    predicted_states = predicted_states_norm * (target_std * 3.0) + target_mean
                else:
                    predicted_states = predicted_states_norm
                
                # 选择第delay_steps步的预测
                step_idx = min(delay_steps, self.prediction_steps - 1)
                predicted_state = predicted_states[0, step_idx, :]  # [4] - 只取环境0
                predicted_roll = predicted_state[0]
                predicted_pitch = predicted_state[1]
                
                # 获取实际状态（只使用环境0）
                actual_roll_env0 = actual_roll[0] if actual_roll.dim() > 0 else actual_roll
                actual_pitch_env0 = actual_pitch[0] if actual_pitch.dim() > 0 else actual_pitch
                
                # 计算预测误差
                orientation_error = torch.sqrt(
                    (predicted_roll - actual_roll_env0)**2 +
                    (predicted_pitch - actual_pitch_env0)**2 + 1e-8
                )
                
                # 计算误差
                mean_error = orientation_error.item()
                evaluation_errors.append(mean_error)
        
        # 如果评估样本不足，返回无穷大
        if len(evaluation_errors) < 5:
            return float('inf')
        
        # 返回平均误差
        return sum(evaluation_errors) / len(evaluation_errors)

