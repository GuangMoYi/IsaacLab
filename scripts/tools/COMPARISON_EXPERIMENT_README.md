# 对比实验脚本使用说明

## 概述

`compare_methods.py` 脚本用于对比不同方法在平台跟随任务上的性能。支持对比以下方法：

- **React-PPO**: 标准强化学习基线
- **React-MPC**: 基于模型的反应式控制基线
- **Oracle-PPO**: 性能上界
- **Ours w/o Prediction**: 消融版本（无预测器）
- **Ours**: 本文提出的方法

## 使用方法

### 基本用法

```bash
python scripts/tools/compare_methods.py \
    --data_dirs \
        "React-PPO:/path/to/react_ppo_data" \
        "React-MPC:/path/to/react_mpc_data" \
        "Oracle-PPO:/path/to/oracle_ppo_data" \
        "Ours w/o Prediction:/path/to/ours_wo_prediction_data" \
        "Ours:/path/to/ours_data" \
    --output_dir /path/to/comparison_results \
    --max_episode_length 1000.0
```

### 参数说明

- `--data_dirs`: 方法数据目录列表，格式为 `方法名:数据目录路径`
  - 例如：`"Ours:/home/user/IsaacLab/training_data"`
  - 可以指定多个方法，每个方法一行或空格分隔
  
- `--output_dir`: 输出目录，用于保存对比结果（默认：`/home/user/IsaacLab/comparison_results`）

- `--max_episode_length`: 最大episode长度，用于计算存活率（默认：1000.0）

### 示例

```bash
# 对比所有方法
python scripts/tools/compare_methods.py \
    --data_dirs \
        "React-PPO:/home/user/IsaacLab/training_data/react_ppo" \
        "React-MPC:/home/user/IsaacLab/training_data/react_mpc" \
        "Oracle-PPO:/home/user/IsaacLab/training_data/oracle_ppo" \
        "Ours w/o Prediction:/home/user/IsaacLab/training_data/ours_wo_prediction" \
        "Ours:/home/user/IsaacLab/training_data/ours" \
    --output_dir /home/user/IsaacLab/comparison_results

# 只对比部分方法
python scripts/tools/compare_methods.py \
    --data_dirs \
        "React-PPO:/path/to/react_ppo" \
        "Ours:/path/to/ours" \
    --output_dir /path/to/results
```

## 数据要求

每个方法的数据目录应包含以下文件：

1. **时间序列数据文件**：
   - `time_series_*.csv` 或 `time_series_*.npz`
   - 应包含以下字段：
     - `time`: 时间（秒）
     - `base_error_ratio`: 基座误差比值
     - `platform_roll`: 平台横滚角（rad）
     - `robot_roll`: 机器狗横滚角（rad）
     - `control_error`: 控制误差（rad）
     - `control_rmse`: 控制误差RMSE（rad）
     - `energy_consumption`: 能量消耗
     - `prediction_error`: 预测误差（rad，仅Ours方法需要）
     - `prediction_rmse`: 预测误差RMSE（rad，仅Ours方法需要）

2. **统计数据文件**（可选）：
   - `training_statistics.npz` 或 `training_statistics.json`
   - 应包含以下字段：
     - `avg_survival_time`: 平均存活时间（步数）
     - `avg_energy_consumption`: 平均能量消耗
     - `max_episode_length`: 最大episode长度（可选）

## 输出结果

脚本会在输出目录中生成以下文件：

### 对比图表（所有方法）

1. **`comparison_base_error_ratio_mse.png`**
   - 各方法基座误差比值的MSE随时间步的变化
   - 使用滑动窗口计算MSE

2. **`comparison_roll_angle.png`**
   - 各方法机器狗横滚角和平台横滚角随时间变化曲线对比
   - 包含平台真实运动（黑色粗线）和各方法的机器狗运动

3. **`comparison_metrics.png`**
   - 各方法的指标对比柱状图
   - 包含三个子图：
     - 控制误差RMSE对比
     - 存活率对比
     - 能量消耗对比

### Ours方法专用图表

4. **`ours_prediction_error_over_time.png`**
   - 平台预测误差随时间的变化
   - 如果数据中包含按预测步长分组的数据，将绘制多条曲线

5. **`ours_prediction_control_correlation.png`**
   - 平台预测误差与机器狗基座和平台的控制误差相关性散点图
   - 包含线性拟合线和相关系数

### 数据表格

6. **`comparison_metrics.csv`**
   - 各方法的指标对比表格（CSV格式）

7. **`comparison_metrics.tex`**
   - 各方法的指标对比表格（LaTeX格式，可用于论文）

## 指标说明

### 计算的指标

1. **控制误差RMSE** (`control_rmse`)
   - 机器狗与平台的姿态误差的均方根误差
   - 单位：rad
   - 越小越好

2. **存活率** (`survival_rate`)
   - 存活率 = 平均存活时间 / 最大episode长度
   - 范围：[0, 1]
   - 越大越好

3. **平均能量消耗** (`avg_energy`)
   - 机器狗的平均能量消耗
   - 越小越好

4. **基座误差比值MSE** (`base_error_ratio_mse`)
   - 基座误差比值的均方误差
   - 越小越好

## 注意事项

1. **数据格式**：确保所有方法的数据格式一致，包含必要的字段

2. **时间对齐**：如果不同方法的数据时间范围不同，脚本会自动处理

3. **缺失数据**：如果某个方法缺少某些数据，脚本会跳过相应的图表生成，并打印警告信息

4. **存活率计算**：如果统计数据中没有 `max_episode_length`，将使用命令行参数的值（默认1000.0）

5. **预测误差数据**：只有Ours方法需要预测误差数据，其他方法不需要

## 故障排除

### 问题1：找不到数据文件

**错误信息**：`[警告] 在 xxx 中没有找到时间序列数据文件`

**解决方案**：
- 检查数据目录路径是否正确
- 确认数据目录中存在 `time_series_*.csv` 或 `time_series_*.npz` 文件

### 问题2：数据字段缺失

**错误信息**：`[警告] xxx方法数据中缺少xxx字段`

**解决方案**：
- 检查数据文件是否包含必要的字段
- 对于对比图表，至少需要 `base_error_ratio` 和 `robot_roll`、`platform_roll`
- 对于Ours方法的专用图表，需要 `prediction_error` 和 `control_error`

### 问题3：存活率为0

**可能原因**：
- 统计数据中没有 `avg_survival_time`
- `max_episode_length` 设置不正确

**解决方案**：
- 检查统计数据文件
- 调整 `--max_episode_length` 参数

## 扩展功能

如果需要添加新的对比指标或图表，可以修改 `compare_methods.py` 脚本：

1. 在 `compute_metrics()` 函数中添加新指标的计算
2. 创建新的绘图函数
3. 在 `main()` 函数中调用新的绘图函数

## 联系与支持

如有问题或建议，请联系开发团队。

