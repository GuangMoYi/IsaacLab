# 绘图数据组合功能使用指南

## 功能说明

在绘图时，可以对数据进行组合计算，例如：
- `control_error * 0.5 + prediction_error`：控制误差乘以0.5再加上预测误差
- `robot_roll - platform_roll * 2`：机器狗横滚角减去平台横滚角乘以2
- `base_error_ratio / 1000 + control_error`：基座误差比值除以1000再加上控制误差

## 使用方法

### 基本语法

使用 `--plot_config` 参数指定数据组合表达式：

```bash
python scripts/tools/compare_methods.py \
    --data_dirs "Ours:/home/user/IsaacLab/training_data" \
    --plot_config '{"base_error_ratio_mse": {"Ours": {"expression": "base_error_ratio * 0.5 + control_error"}}}' \
    --output_dir /home/user/IsaacLab/comparison_results
```

### JSON格式

```json
{
    "图表名称": {
        "方法名": {
            "expression": "数据组合表达式"
        }
    }
}
```

### 支持的图表

1. **base_error_ratio_mse**: 基座误差比值的MSE随时间步的变化
   - 配置键：`"base_error_ratio_mse"`
   - 表达式字段：`"expression"`

2. **roll_angle**: 机器狗横滚角和平台横滚角随时间变化曲线对比
   - 配置键：`"roll_angle"`
   - 表达式字段：`"robot_expression"`（用于机器狗横滚角）

## 表达式语法

### 支持的运算符

- `+` : 加法
- `-` : 减法
- `*` : 乘法
- `/` : 除法

### 支持的列名

所有DataFrame中的列名都可以使用，常见的有：
- `control_error`: 控制误差
- `prediction_error`: 预测误差
- `base_error_ratio`: 基座误差比值
- `robot_roll`: 机器狗横滚角
- `platform_roll`: 平台横滚角
- `robot_pitch`: 机器狗俯仰角
- `platform_pitch`: 平台俯仰角
- `energy_consumption`: 能量消耗
- `time`: 时间

### 表达式示例

1. **简单组合**：
   ```json
   "base_error_ratio * 0.5 + control_error"
   ```

2. **加权组合**：
   ```json
   "control_error * 0.7 + prediction_error * 0.3"
   ```

3. **差值计算**：
   ```json
   "robot_roll - platform_roll"
   ```

4. **缩放后组合**：
   ```json
   "base_error_ratio / 1000 + control_error * 1000"
   ```

5. **多步运算**：
   ```json
   "control_error * 0.5 - prediction_error * 0.3 + base_error_ratio"
   ```

## 完整使用示例

### 示例1：基座误差比值MSE图使用组合数据

```bash
python scripts/tools/compare_methods.py \
    --data_dirs \
        "React-PPO:/path/to/react_ppo" \
        "Ours:/path/to/ours" \
    --plot_config '{
        "base_error_ratio_mse": {
            "Ours": {
                "expression": "base_error_ratio * 0.5 + control_error"
            },
            "React-PPO": {
                "expression": "base_error_ratio * 0.3 + control_error * 0.7"
            }
        }
    }' \
    --output_dir /home/user/IsaacLab/comparison_results
```

### 示例2：横滚角对比图使用组合数据

```bash
python scripts/tools/compare_methods.py \
    --data_dirs \
        "Ours:/path/to/ours" \
    --plot_config '{
        "roll_angle": {
            "Ours": {
                "robot_expression": "robot_roll * 0.8 + platform_roll * 0.2"
            }
        }
    }' \
    --output_dir /home/user/IsaacLab/comparison_results
```

### 示例3：多个图表同时配置

```bash
python scripts/tools/compare_methods.py \
    --data_dirs \
        "React-PPO:/path/to/react_ppo" \
        "Ours:/path/to/ours" \
    --plot_config '{
        "base_error_ratio_mse": {
            "Ours": {
                "expression": "base_error_ratio * 0.5 + control_error"
            }
        },
        "roll_angle": {
            "Ours": {
                "robot_expression": "robot_roll - platform_roll * 0.1"
            }
        }
    }' \
    --output_dir /home/user/IsaacLab/comparison_results
```

## 实际应用场景

### 场景1：加权误差组合

将多个误差指标按权重组合：

```json
{
    "base_error_ratio_mse": {
        "Ours": {
            "expression": "control_error * 0.6 + prediction_error * 0.4"
        }
    }
}
```

### 场景2：相对误差

计算相对误差（相对于平台运动）：

```json
{
    "roll_angle": {
        "Ours": {
            "robot_expression": "robot_roll - platform_roll"
        }
    }
}
```

### 场景3：归一化组合

将不同量纲的数据归一化后组合：

```json
{
    "base_error_ratio_mse": {
        "Ours": {
            "expression": "base_error_ratio / 1000 + control_error * 1000"
        }
    }
}
```

### 场景4：滤波后组合

先对数据进行平滑处理（需要在代码中添加），然后组合：

```python
# 在代码中添加移动平均
df['control_error_smooth'] = df['control_error'].rolling(window=10).mean()
df['prediction_error_smooth'] = df['prediction_error'].rolling(window=10).mean()
```

然后使用：
```json
{
    "base_error_ratio_mse": {
        "Ours": {
            "expression": "control_error_smooth * 0.5 + prediction_error_smooth * 0.5"
        }
    }
}
```

## 注意事项

1. **列名匹配**：确保表达式中的列名在DataFrame中存在
2. **数据类型**：表达式计算结果应该是数值类型
3. **除零检查**：避免除以0的情况
4. **表达式语法**：使用标准的Python数学表达式语法
5. **方法名匹配**：确保配置中的方法名与 `--data_dirs` 中的方法名一致

## 调试技巧

如果表达式没有生效：

1. **检查列名**：打印DataFrame的列名
   ```python
   print(df.columns.tolist())
   ```

2. **测试表达式**：在Python中直接测试表达式
   ```python
   import pandas as pd
   df = pd.read_csv('time_series_0000.csv')
   result = df['control_error'] * 0.5 + df['prediction_error']
   print(result.head())
   ```

3. **查看错误信息**：脚本会打印表达式计算失败的错误信息

## 扩展功能

如果需要更复杂的表达式（如函数调用、条件判断等），可以：

1. 在 `compute_plot_data` 函数中添加更多支持
2. 在数据加载后预处理，创建新的列
3. 使用自定义函数

例如，添加绝对值函数：

```python
def compute_plot_data(df: pd.DataFrame, expression: str):
    # ... 现有代码 ...
    
    # 支持 abs() 函数
    expr_code = expr_code.replace('abs(', 'np.abs(')
    
    # ... 其余代码 ...
```

然后在表达式中使用：
```json
{
    "base_error_ratio_mse": {
        "Ours": {
            "expression": "abs(control_error) * 0.5 + abs(prediction_error) * 0.5"
        }
    }
}
```

