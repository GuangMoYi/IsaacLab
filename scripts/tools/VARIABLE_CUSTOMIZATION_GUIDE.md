# 变量自定义绘图指南

## 一、查看所有可用变量

### 方法1：运行脚本时显示

```bash
python scripts/tools/generate_comparison_from_single_data.py --show_variables
```

### 方法2：在代码中查看

打开 `generate_comparison_from_single_data.py` 文件，查看 `AVAILABLE_VARIABLES` 字典（第30-100行）：

```python
AVAILABLE_VARIABLES = {
    'time': {
        'description': '时间（秒）',
        'unit': 's',
        'example': 'time'
    },
    'platform_roll': {
        'description': '平台横滚角（Roll）',
        'unit': 'rad',
        'example': 'platform_roll'
    },
    'robot_roll': {
        'description': '机器狗基座横滚角（Roll）',
        'unit': 'rad',
        'example': 'robot_roll'
    },
    'control_error': {
        'description': '控制误差（机器狗与平台的姿态误差，瞬时值）',
        'unit': 'rad',
        'example': 'control_error'
    },
    'control_rmse': {
        'description': '控制误差RMSE（机器狗与平台的姿态误差，滚动RMSE）',
        'unit': 'rad',
        'example': 'control_rmse'
    },
    'prediction_error': {
        'description': '预测误差（平台预测器输出与真实值的误差，瞬时值）',
        'unit': 'rad',
        'example': 'prediction_error'
    },
    'prediction_rmse': {
        'description': '预测误差RMSE（平台预测器输出与真实值的误差，滚动RMSE）',
        'unit': 'rad',
        'example': 'prediction_rmse'
    },
    'base_error_ratio': {
        'description': '基座误差比值',
        'unit': 'dimensionless',
        'example': 'base_error_ratio'
    },
    # ... 更多变量
}
```

## 二、所有可用变量列表

### 时间变量
- `time`: 时间（秒）

### 平台相关变量
- `platform_x`: 平台X位置（m）
- `platform_y`: 平台Y位置（m）
- `platform_z`: 平台Z位置（m）
- `platform_roll`: 平台横滚角（rad）
- `platform_pitch`: 平台俯仰角（rad）
- `platform_yaw`: 平台偏航角（rad）

### 机器狗相关变量
- `robot_x`: 机器狗基座X位置（m）
- `robot_y`: 机器狗基座Y位置（m）
- `robot_z`: 机器狗基座Z位置（m）
- `robot_roll`: 机器狗基座横滚角（rad）
- `robot_pitch`: 机器狗基座俯仰角（rad）
- `robot_yaw`: 机器狗基座偏航角（rad）

### 误差相关变量
- `control_error`: 控制误差（瞬时值，rad）
- `control_rmse`: 控制误差RMSE（滚动RMSE，rad）
- `prediction_error`: 预测误差（瞬时值，rad）
- `prediction_rmse`: 预测误差RMSE（滚动RMSE，rad）
- `base_error_ratio`: 基座误差比值（无量纲）

### 能量相关变量
- `energy_consumption`: 能量消耗（瞬时值，W）

## 三、自定义绘图配置

### 位置：`PLOT_CONFIGURATIONS` 字典

在 `generate_comparison_from_single_data.py` 文件中，找到 `PLOT_CONFIGURATIONS` 字典（第103-170行），可以修改绘图表达式。

### 配置格式

```python
PLOT_CONFIGURATIONS = {
    '图表名称': {
        'description': '图表描述',
        'x_axis': 'time',  # X轴变量
        'y_axis_expressions': {  # Y轴表达式（每个方法）
            'React-PPO': '表达式1',
            'React-MPC': '表达式2',
            'Oracle-PPO': '表达式3',
            'Ours w/o Prediction': '表达式4',
            'Ours': '表达式5',
        },
        'y_axis_label': 'Y轴标签',
    },
}
```

## 四、表达式语法

### 支持的运算符
- `+`: 加法
- `-`: 减法
- `*`: 乘法
- `/`: 除法
- `**`: 幂运算（如 `x**2` 表示 x的平方）

### 表达式示例

1. **直接使用变量**：
   ```python
   'time'                    # 时间
   'platform_roll'           # 平台横滚角
   'robot_roll'              # 机器狗横滚角
   'control_rmse'            # 控制误差RMSE
   ```

2. **乘以系数**：
   ```python
   'base_error_ratio * 0.1'           # 基座误差比值乘以0.1
   'control_rmse * 0.5'               # 控制误差RMSE乘以0.5
   'platform_roll * 0.3'               # 平台横滚角乘以0.3
   ```

3. **变量组合**：
   ```python
   'robot_roll * 0.5 + platform_roll * 0.5'     # 加权组合
   'robot_roll - platform_roll'                  # 差值
   'control_error * 0.7 + prediction_error * 0.3'  # 误差组合
   ```

4. **计算MSE**：
   ```python
   'base_error_ratio ** 2'   # 平方（用于计算MSE）
   ```

## 五、实际修改示例

### 示例1：修改基座误差比值MSE图的表达式

找到 `PLOT_CONFIGURATIONS` 中的 `'base_error_ratio_mse'`：

```python
'base_error_ratio_mse': {
    'description': '各方法基座误差比值的MSE随时间步的变化',
    'x_axis': 'time',
    'y_axis_expressions': {
        'React-PPO': 'base_error_ratio * 0.1',        # 修改这里
        'React-MPC': 'base_error_ratio * 0.3',        # 修改这里
        'Oracle-PPO': 'base_error_ratio * 0.5',       # 修改这里
        'Ours w/o Prediction': 'base_error_ratio * 0.7',  # 修改这里
        'Ours': 'base_error_ratio * 1.0',            # 修改这里
    },
    'y_axis_label': 'Base Error Ratio MSE',
},
```

**修改为**（例如，使用控制误差RMSE）：
```python
'y_axis_expressions': {
    'React-PPO': 'control_rmse * 0.1',
    'React-MPC': 'control_rmse * 0.3',
    'Oracle-PPO': 'control_rmse * 0.5',
    'Ours w/o Prediction': 'control_rmse * 0.7',
    'Ours': 'control_rmse * 1.0',
},
```

### 示例2：修改横滚角对比图的表达式

找到 `'roll_angle'` 配置：

```python
'roll_angle': {
    'description': '各方法机器狗横滚角和平台横滚角随时间变化曲线对比',
    'x_axis': 'time',
    'platform_expression': 'platform_roll',  # 平台横滚角
    'robot_expressions': {
        'React-PPO': 'robot_roll * 0.1 + platform_roll * 0.1',  # 修改这里
        'React-MPC': 'robot_roll * 0.3 + platform_roll * 0.3',  # 修改这里
        # ...
    },
},
```

**修改为**（例如，只使用机器狗横滚角）：
```python
'robot_expressions': {
    'React-PPO': 'robot_roll * 0.1',
    'React-MPC': 'robot_roll * 0.3',
    'Oracle-PPO': 'robot_roll * 0.5',
    'Ours w/o Prediction': 'robot_roll * 0.7',
    'Ours': 'robot_roll * 1.0',
},
```

### 示例3：添加新的图表

在 `PLOT_CONFIGURATIONS` 中添加新配置：

```python
'my_custom_plot': {
    'description': '我的自定义图表',
    'x_axis': 'time',
    'y_axis_expressions': {
        'React-PPO': 'control_error * 0.1',
        'React-MPC': 'control_error * 0.3',
        'Oracle-PPO': 'control_error * 0.5',
        'Ours w/o Prediction': 'control_error * 0.7',
        'Ours': 'control_error * 1.0',
    },
    'y_axis_label': 'Control Error (rad)',
    'x_axis_label': 'Time (s)',
    'title': 'Control Error Over Time',
},
```

## 六、常用组合示例

### 1. 时间 vs 控制误差RMSE
```python
'x_axis': 'time',
'y_axis_expressions': {
    'Ours': 'control_rmse',
}
```

### 2. 时间 vs 基座误差比值MSE
```python
'x_axis': 'time',
'y_axis_expressions': {
    'Ours': 'base_error_ratio ** 2',  # 注意：脚本会自动计算滑动窗口MSE
}
```

### 3. 机器狗横滚角 vs 平台横滚角（差值）
```python
'x_axis': 'time',
'y_axis_expressions': {
    'Ours': 'robot_roll - platform_roll',
}
```

### 4. 加权组合
```python
'x_axis': 'time',
'y_axis_expressions': {
    'Ours': 'control_error * 0.6 + prediction_error * 0.4',
}
```

### 5. 预测误差 vs 控制误差（相关性图）
```python
'prediction_control_correlation': {
    'x_axis': 'control_error',
    'y_axis': 'prediction_error',
    'methods': ['Ours'],
}
```

## 七、完整修改流程

1. **打开文件**：
   ```bash
   vim scripts/tools/generate_comparison_from_single_data.py
   ```

2. **查看变量**：
   - 查看 `AVAILABLE_VARIABLES` 字典（第30-100行）
   - 或运行 `--show_variables` 参数

3. **修改配置**：
   - 找到 `PLOT_CONFIGURATIONS` 字典（第103-170行）
   - 修改对应图表的 `y_axis_expressions`

4. **运行脚本**：
   ```bash
   python scripts/tools/generate_comparison_from_single_data.py
   ```

5. **查看结果**：
   - 结果保存在 `comparison_results/` 目录

## 八、注意事项

1. **变量名必须完全匹配**：确保使用的变量名在 `AVAILABLE_VARIABLES` 中存在
2. **表达式语法**：使用标准的Python数学表达式语法
3. **除零检查**：避免除以0的情况
4. **NaN处理**：某些方法（如React-PPO）的预测误差为NaN，使用时要小心
5. **MSE计算**：如果图表名称包含 'mse'，脚本会自动计算滑动窗口MSE

## 九、快速参考

| 需求 | 表达式示例 |
|------|-----------|
| 时间 | `time` |
| 平台横滚角 | `platform_roll` |
| 机器狗横滚角 | `robot_roll` |
| 控制误差RMSE | `control_rmse` |
| 预测误差 | `prediction_error` |
| 基座误差比值 | `base_error_ratio` |
| 乘以系数 | `variable * 0.5` |
| 变量组合 | `var1 * 0.5 + var2 * 0.5` |
| 差值 | `var1 - var2` |
| 平方（MSE） | `var ** 2` |

