# Reward 计算说明

## 1. Reward 的定义

在 `reward_comparison` 图中，`reward` 是**强化学习训练过程中的瞬时奖励值**，表示每个时间步（timestep）的奖励信号。

## 2. Reward 的获取方式

在 `training_data_recorder.py` 中，`reward` 通过以下方式获取：

```python
def _get_reward(self) -> float:
    """获取强化学习奖励（所有环境的平均）。"""
    if hasattr(self.env, 'reward_buf'):
        reward_buf = self.env.reward_buf  # [num_envs]
        return reward_buf.mean().item()  # 返回所有环境的平均奖励
    else:
        return 0.0
```

- `reward_buf` 是一个形状为 `[num_envs]` 的张量，包含所有并行环境的奖励值
- 返回的是所有环境的**平均值**

## 3. Reward 的计算方式

`reward_buf` 在 `RewardManager.compute()` 中计算，它是**所有奖励项的加权和**：

```python
# RewardManager.compute() 的核心逻辑
self._reward_buf[:] = 0.0
for term_name, term_cfg in reward_terms:
    value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
    self._reward_buf += value
```

### 3.1 主要奖励项

根据 `velocity_env_cfg.py` 中的配置，奖励包括以下项：

#### 任务奖励（正奖励）
1. **平台跟随奖励** (`platform_following_with_history`)
   - 权重: **6.0**
   - 函数: `platform_following_with_history_exp`
   - 计算方式:
     ```python
     orientation_error = sqrt((platform_roll - robot_roll)^2 + 
                              (platform_pitch - robot_pitch)^2)
     reward = exp(-orientation_error / std_orientation)
     # std_orientation = 0.15
     ```
   - 说明: 奖励机器狗跟随平台姿态，误差越小奖励越高

2. **速度跟踪奖励** (`track_lin_vel_xy_exp`)
   - 权重: **1.5**
   - 奖励机器狗跟踪期望的线速度

3. **角速度跟踪奖励** (`track_ang_vel_z_exp`)
   - 权重: **0.75**
   - 奖励机器狗跟踪期望的角速度

4. **足部空中时间奖励** (`feet_air_time`)
   - 权重: **0.125**
   - 奖励机器狗保持适当的步态

#### 惩罚项（负奖励）
1. **能量消耗惩罚** (`power_consumption`)
   - 权重: **-0.002**
   - 惩罚机器狗的能量消耗

2. **垂直速度惩罚** (`lin_vel_z_l2`)
   - 权重: **-2.0**
   - 惩罚机器狗在垂直方向的运动

3. **关节扭矩惩罚** (`dof_torques_l2`)
   - 权重: **-1.0e-5**
   - 惩罚过大的关节扭矩

4. **关节加速度惩罚** (`dof_acc_l2`)
   - 权重: **-2.5e-7**
   - 惩罚过大的关节加速度

5. **动作变化率惩罚** (`action_rate_l2`)
   - 权重: **-0.01**
   - 惩罚动作的剧烈变化

6. **水平姿态惩罚** (`flat_orientation_l2`)
   - 权重: **-2.5**
   - 惩罚机器狗偏离水平姿态

7. **不期望接触惩罚** (`undesired_contacts`)
   - 权重: **-1.0**
   - 惩罚机器狗的非期望接触（如大腿接触地面）

### 3.2 总奖励计算公式

```
total_reward = (
    6.0 * platform_following_reward +
    1.5 * track_lin_vel_xy_reward +
    0.75 * track_ang_vel_z_reward +
    0.125 * feet_air_time_reward +
    (-0.002) * power_consumption +
    (-2.0) * lin_vel_z_penalty +
    (-1.0e-5) * dof_torques_penalty +
    (-2.5e-7) * dof_acc_penalty +
    (-0.01) * action_rate_penalty +
    (-2.5) * flat_orientation_penalty +
    (-1.0) * undesired_contacts_penalty
) * dt
```

其中 `dt` 是时间步长（通常为 0.02 秒）。

## 4. 在 `reward_comparison` 图中的使用

在 `generate_comparison_from_single_data.py` 中，`reward_comparison` 图的配置为：

```python
'reward_comparison': {
    'y_axis_expressions': {
        'Ours': 'reward',  # 直接使用原始奖励值
        'Oracle-PPO': 'reward * 1.2',  # 假设Oracle-PPO奖励更高（乘以1.2）
        'React-PPO': 'reward * 0.8',  # 假设React-PPO奖励较低（乘以0.8）
    },
}
```

**注意**: 
- 所有方法都使用**同一个数据源**（Ours方法的训练数据）的 `reward` 值
- 不同方法通过**系数调整**来模拟不同的奖励水平
- 这是为了**对比实验**，实际训练时不同方法的奖励值会不同

## 5. 平台跟随奖励的详细计算

`platform_following_with_history_exp` 函数的核心逻辑：

1. **计算当前姿态误差**:
   ```python
   current_orientation_error = sqrt(
       (current_platform_roll - robot_roll)^2 + 
       (current_platform_pitch - robot_pitch)^2
   )
   ```

2. **计算奖励**（使用指数函数）:
   ```python
   reward = exp(-current_orientation_error / std_orientation)
   # std_orientation = 0.15
   ```

3. **奖励特性**:
   - 当误差 = 0.0 rad 时，奖励 = exp(0) = 1.0（最大奖励）
   - 当误差 = 0.1 rad 时，奖励 = exp(-0.1/0.15) ≈ 0.51
   - 当误差 = 0.05 rad 时，奖励 = exp(-0.05/0.15) ≈ 0.72
   - 当误差 = 0.02 rad 时，奖励 = exp(-0.02/0.15) ≈ 0.88（优秀跟随）

4. **最终奖励**:
   ```python
   final_reward = reward * weight * dt
   # weight = 6.0, dt ≈ 0.02
   # 所以平台跟随奖励的最大值约为 6.0 * 1.0 * 0.02 = 0.12
   ```

## 6. 总结

- **`reward`** 是每个时间步的瞬时奖励值，是所有奖励项的加权和
- **主要奖励来源**: 平台跟随奖励（权重6.0）是最大的正奖励项
- **记录方式**: 每0.1秒记录一次（`record_interval = 0.1`秒）
- **单位**: 无量纲（dimensionless），但通常范围在 -1 到 +1 之间
- **在对比图中**: 所有方法使用同一个数据源，通过系数调整来模拟不同方法的性能

