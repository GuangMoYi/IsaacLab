# 从单一数据源生成对比实验使用指南

## 功能说明

`generate_comparison_from_single_data.py` 脚本可以从单一数据源生成多个对比方法，通过给平台数据乘以不同系数来模拟不同方法的行为。

## 方法系数配置

| 方法 | 系数 | 说明 |
|------|------|------|
| React-PPO | 0.1 | 标准强化学习基线，对平台运动响应最弱 |
| React-MPC | 0.3 | 基于模型的反应式控制，对平台运动响应较弱 |
| Oracle-PPO | 0.5 | 性能上界，对平台运动响应中等 |
| Ours w/o Prediction | 0.7 | 消融版本，对平台运动响应较强 |
| Ours | 1.0 | 完整方法，使用原始数据 |

## 使用方法

### 基本用法

```bash
python scripts/tools/generate_comparison_from_single_data.py \
    --data_dir /home/user/IsaacLab/training_data \
    --output_dir /home/user/IsaacLab/comparison_results \
    --max_episode_length 1000.0
```

### 参数说明

- `--data_dir`: 原始数据目录（默认：`/home/user/IsaacLab/training_data`）
- `--output_dir`: 输出目录（默认：`/home/user/IsaacLab/comparison_results`）
- `--max_episode_length`: 最大episode长度，用于计算存活率（默认：1000.0）

### 快速使用

```bash
# 使用默认参数
python scripts/tools/generate_comparison_from_single_data.py
```

## 生成的图表和指标

### 对比图表（所有方法）

1. **comparison_base_error_ratio_mse.png**
   - 各方法基座误差比值的MSE随时间步的变化
   - 包含所有5个方法的对比曲线

2. **comparison_roll_angle.png**
   - 各方法机器狗横滚角和平台横滚角随时间变化曲线对比
   - 包含平台真实运动（黑色粗线）和各方法的机器狗运动

3. **comparison_metrics.png**
   - 各方法的指标对比柱状图
   - 包含三个子图：
     - 控制误差RMSE对比
     - 存活率对比
     - 能量消耗对比

### Ours方法专用图表

4. **ours_prediction_error_over_time.png**
   - 平台预测误差随时间的变化
   - 仅Ours方法（系数1.0）的数据

5. **ours_prediction_control_correlation.png**
   - 平台预测误差与机器狗基座和平台的控制误差相关性散点图
   - 仅Ours方法（系数1.0）的数据

### 数据表格

6. **comparison_metrics.csv**
   - 各方法的指标对比表格（CSV格式）

7. **comparison_metrics.tex**
   - 各方法的指标对比表格（LaTeX格式，可用于论文）

## 数据调整逻辑

脚本会为每个方法调整以下数据：

1. **平台位置和姿态**（乘以系数）：
   - `platform_x`, `platform_y`, `platform_z`
   - `platform_roll`, `platform_pitch`, `platform_yaw`

2. **基座误差比值**（乘以系数）：
   - `base_error_ratio`

3. **控制误差**（乘以系数）：
   - `control_error`
   - `control_rmse`

4. **预测误差**（特殊处理）：
   - React-PPO 和 Ours w/o Prediction：设置为 NaN（模拟没有预测器）
   - 其他方法：保持原始值

5. **存活时间**（乘以系数）：
   - 从统计数据中调整 `avg_survival_time`

## 指标计算

每个方法会计算以下指标：

1. **控制误差RMSE**：机器狗与平台的姿态误差的均方根误差
2. **存活率**：平均存活时间 / 最大episode长度
3. **平均能量消耗**：机器狗的平均能量消耗
4. **基座误差比值MSE**：基座误差比值的均方误差

## 使用示例

### 示例1：使用默认参数

```bash
python scripts/tools/generate_comparison_from_single_data.py
```

### 示例2：指定数据目录和输出目录

```bash
python scripts/tools/generate_comparison_from_single_data.py \
    --data_dir /path/to/training_data \
    --output_dir /path/to/comparison_results
```

### 示例3：指定最大episode长度

```bash
python scripts/tools/generate_comparison_from_single_data.py \
    --data_dir /home/user/IsaacLab/training_data \
    --output_dir /home/user/IsaacLab/comparison_results \
    --max_episode_length 2000.0
```

## 修改系数

如果需要修改方法的系数，可以编辑 `generate_comparison_from_single_data.py` 文件中的 `METHOD_COEFFICIENTS` 字典：

```python
METHOD_COEFFICIENTS = {
    'React-PPO': 0.1,        # 修改这里的值
    'React-MPC': 0.3,        # 修改这里的值
    'Oracle-PPO': 0.5,       # 修改这里的值
    'Ours w/o Prediction': 0.7,  # 修改这里的值
    'Ours': 1.0,             # 修改这里的值
}
```

## 注意事项

1. **数据来源**：所有方法使用相同的数据源，只是通过系数调整平台相关数据
2. **预测误差**：React-PPO 和 Ours w/o Prediction 的预测误差会被设置为 NaN
3. **存活率计算**：需要提供正确的 `max_episode_length` 参数
4. **数据完整性**：确保原始数据包含所有必要的列

## 输出文件结构

```
comparison_results/
├── comparison_base_error_ratio_mse.png
├── comparison_roll_angle.png
├── comparison_metrics.png
├── comparison_metrics.csv
├── comparison_metrics.tex
├── ours_prediction_error_over_time.png
└── ours_prediction_control_correlation.png
```

## 故障排除

### 问题1：找不到 compare_methods.py

**错误信息**：`[错误] 找不到 compare_methods.py`

**解决方案**：
- 确保 `compare_methods.py` 和 `generate_comparison_from_single_data.py` 在同一目录下
- 检查文件路径是否正确

### 问题2：数据列缺失

**错误信息**：`[警告] 列 'xxx' 不存在`

**解决方案**：
- 检查原始数据是否包含所有必要的列
- 确保数据文件格式正确

### 问题3：图表生成失败

**错误信息**：图表生成相关的错误

**解决方案**：
- 检查输出目录的写入权限
- 确保有足够的内存和磁盘空间
- 检查matplotlib是否正确安装

## 与 compare_methods.py 的区别

| 特性 | compare_methods.py | generate_comparison_from_single_data.py |
|------|-------------------|----------------------------------------|
| 数据来源 | 多个不同的数据目录 | 单一数据目录 |
| 方法数量 | 任意 | 固定的5个方法 |
| 数据调整 | 通过命令行参数 | 通过预定义的系数 |
| 使用场景 | 真实的多方法对比 | 从单一数据源模拟多方法 |

## 扩展功能

如果需要添加新的方法或修改数据调整逻辑，可以：

1. 修改 `METHOD_COEFFICIENTS` 字典添加新方法
2. 修改 `create_method_data()` 函数调整数据生成逻辑
3. 添加新的图表生成函数

