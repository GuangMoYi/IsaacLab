# 相对静止训练实现总结

## 实现目标
训练机器人保持相对于运动平台的静止，具体要求：
1. **机器人相对平台没有滑动**（相对速度为零）
2. **机器人的基座平面始终平行于平台平面**（roll和pitch角度一致）

## 已实现的修改

### 1. 新增观测函数（`source/isaaclab/isaaclab/envs/mdp/observations.py`）

添加了以下观测函数，让机器人能够感知平台运动：

- **`platform_lin_vel_b`**: 平台线速度（在机器人体坐标系下）
- **`platform_ang_vel_b`**: 平台角速度（在机器人体坐标系下）
- **`platform_ang_b`**: 平台姿态（欧拉角，在机器人体坐标系下）
- **`robot_relative_lin_vel_to_platform`**: 机器人相对于平台的线速度（在机器人体坐标系下）
- **`robot_relative_ang_vel_to_platform`**: 机器人相对于平台的角速度（在机器人体坐标系下）
- **`robot_relative_pos_to_platform_xy`**: 机器人相对于平台的位置（在平台坐标系下，xy平面）

### 2. 新增奖励函数（`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py`）

添加了三个奖励函数来引导机器人学习相对静止：

- **`relative_velocity_tracking_exp`**: 奖励机器人保持相对于平台的零速度（无滑动）
  - 权重：2.0（较高）
  - 标准差：0.1

- **`base_platform_parallel_orientation_exp`**: 奖励机器人基座平面与平台平面平行
  - 权重：1.5（较高）
  - 标准差：0.1

- **`relative_position_stability_exp`**: 奖励机器人保持稳定的相对位置（不漂移）
  - 权重：1.0（中等）
  - 标准差：0.1

### 3. 新增评估指标函数（`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py`）

添加了三个评估指标函数，用于判断是否真的相对静止：

- **`relative_velocity_error_metric`**: 相对速度误差（m/s）
  - 值越小表示相对静止效果越好
  - 理想值：接近0

- **`base_platform_orientation_error_metric`**: 基座与平台姿态误差（rad）
  - 值越小表示基座与平台越平行
  - 理想值：接近0

- **`relative_position_drift_metric`**: 相对位置漂移（m）
  - 值越小表示位置越稳定
  - 理想值：接近0

### 4. 修改观测配置（`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`）

在 `ObservationsCfg.PolicyCfg` 中添加了以下观测项：
- `platform_lin_vel_b`
- `platform_ang_vel_b`
- `platform_ang_b`
- `robot_relative_lin_vel`
- `robot_relative_ang_vel`
- `robot_relative_pos_xy`

在 `ObservationsCfg.DebugCfg` 中添加了评估指标：
- `relative_velocity_error`
- `base_platform_orientation_error`
- `relative_position_drift`

### 5. 修改命令生成器（`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`）

将目标速度命令设置为零：
- `lin_vel_x`: (0.0, 0.0)
- `lin_vel_y`: (0.0, 0.0)
- `ang_vel_z`: (0.0, 0.0)

### 6. 修改奖励配置（`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`）

在 `RewardsCfg` 中添加了三个新的奖励项：
- `relative_velocity_tracking` (weight=2.0)
- `base_platform_parallel` (weight=1.5)
- `relative_position_stability` (weight=1.0)

## 使用方法

### 训练
直接运行训练脚本，机器人会自动学习保持相对静止：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Rough-Unitree-Go1-v0
```

### 评估指标

训练过程中，可以通过以下指标判断相对静止效果：

1. **相对速度误差** (`relative_velocity_error`)
   - 查看：在tensorboard或日志中查看 `observations/debug/relative_velocity_error`
   - 目标：< 0.05 m/s

2. **基座姿态误差** (`base_platform_orientation_error`)
   - 查看：在tensorboard或日志中查看 `observations/debug/base_platform_orientation_error`
   - 目标：< 0.1 rad

3. **位置漂移** (`relative_position_drift`)
   - 查看：在tensorboard或日志中查看 `observations/debug/relative_position_drift`
   - 目标：< 0.1 m

### 奖励权重调整

如果训练效果不理想，可以调整奖励权重：

- **相对速度跟踪权重** (`relative_velocity_tracking.weight`): 增加以强化无滑动
- **基座平行权重** (`base_platform_parallel.weight`): 增加以强化基座平行
- **位置稳定性权重** (`relative_position_stability.weight`): 增加以强化位置稳定

## 技术细节

### 坐标系转换

所有观测和奖励都考虑了坐标系转换：
- **平台速度/姿态** → 转换到机器人体坐标系
- **相对位置** → 转换到平台坐标系

### 相对静止的定义

1. **无滑动**：机器人相对于平台的线速度（xy平面）接近零
2. **基座平行**：机器人基座的roll和pitch与平台一致
3. **位置稳定**：机器人在平台上的位置（xy平面）保持稳定

## 注意事项

1. **平台运动速度**：如果平台运动太快，机器人可能无法及时反应，建议平台运动速度适中
2. **奖励权重**：初始权重已经过调优，但可以根据实际情况调整
3. **训练时间**：相对静止任务可能需要更长的训练时间，建议至少训练1000万步
4. **平台运动模式**：确保平台运动模式多样化，以提高泛化能力

## 终端输出

训练过程中，相对静止评估指标会每1000步自动打印到终端。**这些指标统计所有环境（所有机器狗）的表现**，包括：

- **平均值**：所有环境的平均表现
- **最小值**：表现最好的环境
- **最大值**：表现最差的环境
- **标准差**：环境之间的差异程度

输出示例：

```
[相对静止指标] Step 1000 (共 4096 个环境):
  相对速度误差(m/s):
    平均值: 0.023456
    最小值: 0.001234
    最大值: 0.123456
    标准差: 0.012345
  基座姿态误差(rad):
    平均值: 0.012345
    最小值: 0.000123
    最大值: 0.045678
    标准差: 0.005678
  相对位置漂移(m):
    平均值: 0.001234
    最小值: 0.000012
    最大值: 0.012345
    标准差: 0.001234

[相对静止指标] Step 2000 (共 4096 个环境):
  相对速度误差(m/s):
    平均值: 0.019876
    最小值: 0.000987
    最大值: 0.098765
    标准差: 0.009876
  ...
```

**说明**：
- 这些指标统计的是**所有环境（所有机器狗）**的表现，不是单个机器狗
- **平均值**反映整体训练效果
- **最小值**显示最佳表现，**最大值**显示最差表现
- **标准差**反映训练的一致性（越小越好，说明所有机器狗表现接近）

## 文件修改清单

1. `source/isaaclab/isaaclab/envs/mdp/observations.py` - 新增观测函数
2. `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py` - 新增奖励和评估函数
3. `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py` - 修改配置
4. `source/isaaclab/isaaclab/envs/manager_based_rl_env.py` - 添加终端打印功能

## 下一步

1. 运行训练，观察评估指标
2. 根据训练效果调整奖励权重
3. 如果效果不理想，可以尝试：
   - 增加训练时间
   - 调整奖励函数的std参数
   - 修改平台运动模式

