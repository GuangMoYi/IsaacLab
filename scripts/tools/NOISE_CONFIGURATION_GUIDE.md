# 噪声配置指南

本指南说明如何在绘图时给各变量添加高斯噪声。

## 功能说明

在绘图时，可以为每个变量添加高斯噪声，用于：
- 模拟测量噪声
- 测试方法的鲁棒性
- 添加数据增强效果

## 推荐方式：在表达式中直接使用 noise() 函数

**最简单直接的方式**是在表达式中直接使用 `noise(mean, variance)` 函数：

```python
'robot_expressions': {
    'React-PPO': 'robot_roll * 0.6 + platform_roll * 0.4 + noise(0, 0.01)',
    'Ours': 'robot_roll * 0.3 + platform_roll * 0.7 + noise(0, 0.005)',
}
```

### 语法格式

```
noise(mean, variance)
```

- `mean`: 噪声均值（通常为0.0）
- `variance`: 噪声方差（必须大于0才会添加噪声）
  - 注意：第二个参数是**方差**，不是标准差
  - 标准差 = sqrt(方差)
  - 例如：`noise(0, 0.01)` 表示均值0、方差0.01的高斯噪声，对应的标准差为 sqrt(0.01) = 0.1

### 使用示例

```python
# 示例1: 添加小噪声（方差0.01，标准差约为0.1）
'robot_roll + noise(0, 0.01)'

# 示例2: 在组合表达式中添加噪声（方差0.02，标准差约为0.141）
'robot_roll * 0.5 + platform_roll * 0.5 + noise(0, 0.02)'

# 示例3: 添加多个噪声（会累加）
'robot_roll + noise(0, 0.01) + noise(0, 0.005)'

# 示例4: 非零均值噪声（均值0.001，方差0.01）
'robot_roll + noise(0.001, 0.01)'

# 示例5: 在复杂表达式中使用
'sqrt(platform_roll ** 2 + platform_pitch ** 2) + noise(0, 0.01)'
```

## 旧方式：通过配置字典（已废弃，保留以兼容）

### 噪声配置格式

噪声配置使用字典格式，支持以下参数：

```python
{
    'mean': 0.0,    # 噪声均值（默认0.0）
    'std': 0.01,    # 噪声标准差（默认0.0，即不添加噪声）
    'seed': None,   # 随机种子（可选，用于可重复性）
}
```

## 配置方式

### 方式1: 全局默认噪声

在图表配置中添加 `noise_config`，设置全局默认噪声：

```python
'roll_angle': {
    'description': '横滚角对比',
    'robot_expressions': {...},
    'noise_config': {
        'default': {'mean': 0.0, 'std': 0.01, 'seed': 42},
    },
}
```

### 方式2: 方法特定噪声

为不同方法设置不同的噪声：

```python
'roll_angle': {
    'description': '横滚角对比',
    'robot_expressions': {...},
    'noise_config': {
        'default': {'mean': 0.0, 'std': 0.01, 'seed': 42},  # 默认噪声
        'methods': {  # 方法特定噪声（会覆盖默认值）
            'React-PPO': {'mean': 0.0, 'std': 0.02, 'seed': 42},
            'Ours': {'mean': 0.0, 'std': 0.005, 'seed': 42},
        },
    },
}
```

### 方式3: 在表达式配置中直接添加

对于使用 `plot_config` 的绘图函数，可以在方法配置中直接添加 `noise`：

```python
plot_config = {
    'React-PPO': {
        'robot_expression': 'robot_roll * 0.6 + platform_roll * 0.4',
        'noise': {'mean': 0.0, 'std': 0.02, 'seed': 42},
    },
    'Ours': {
        'robot_expression': 'robot_roll * 0.3 + platform_roll * 0.7',
        'noise': {'mean': 0.0, 'std': 0.005, 'seed': 42},
    },
}
```

## 参数说明

### mean (均值)
- **类型**: float
- **默认值**: 0.0
- **说明**: 高斯噪声的均值。通常设置为0.0，表示噪声围绕原始值对称分布。

### std (标准差)
- **类型**: float
- **默认值**: 0.0
- **说明**: 高斯噪声的标准差。值越大，噪声越大。
  - `std = 0.0`: 不添加噪声
  - `std = 0.01`: 小噪声（约1%的标准差）
  - `std = 0.1`: 中等噪声（约10%的标准差）
  - `std = 1.0`: 大噪声（与数据同量级）

### seed (随机种子)
- **类型**: int 或 None
- **默认值**: None
- **说明**: 随机数生成器的种子。设置后可以确保每次运行产生相同的噪声，用于可重复性实验。

## 完整示例

### 示例1: 在横滚角表达式中添加噪声

```python
'roll_angle': {
    'description': '横滚角对比',
    'robot_expressions': {
        'React-PPO': 'robot_roll * 0.6 + platform_roll * 0.4 + noise(0, 0.02)',
        'React-MPC': 'robot_roll * 0.5 + platform_roll * 0.5 + noise(0, 0.01)',
        'Oracle-PPO': 'robot_roll * 0.2 + platform_roll * 0.8 + noise(0, 0.005)',
        'Ours': 'robot_roll * 0.3 + platform_roll * 0.7 + noise(0, 0.01)',
    },
}
```

### 示例2: 在位置数据中添加噪声（单位：米）

```python
'x_position': {
    'description': 'X位置对比（带测量噪声）',
    'robot_expressions': {
        'React-PPO': 'robot_x * 0.6 + platform_x * 0.4 + noise(0, 0.001)',  # 1mm噪声
        'Ours': 'robot_x * 0.3 + platform_x * 0.7 + noise(0, 0.0005)',      # 0.5mm噪声
    },
}
```

### 示例3: 在Y轴表达式中添加噪声

```python
'y_axis_expressions': {
    'React-PPO': 'base_error_ratio * 0.1 + noise(0, 0.01)',
    'Ours': 'base_error_ratio * 1.0 + noise(0, 0.005)',
}
```

## 旧方式：通过配置字典（已废弃，保留以兼容）

### 使用示例（旧方式）

### 示例1: 为所有方法添加相同的小噪声

```python
'pitch_angle': {
    'description': '俯仰角对比',
    'robot_expressions': {...},
    'noise_config': {
        'default': {'mean': 0.0, 'std': 0.01, 'seed': 42},
    },
}
```

### 示例2: 为不同方法添加不同强度的噪声

```python
'roll_angle': {
    'description': '横滚角对比',
    'robot_expressions': {...},
    'noise_config': {
        'default': {'mean': 0.0, 'std': 0.01, 'seed': 42},
        'methods': {
            'React-PPO': {'mean': 0.0, 'std': 0.02, 'seed': 42},  # 较大噪声
            'Ours': {'mean': 0.0, 'std': 0.005, 'seed': 42},      # 较小噪声
        },
    },
}
```

### 示例3: 模拟传感器噪声

假设角度传感器有 ±0.05 rad 的测量误差（3σ原则，std ≈ 0.017）：

```python
'roll_angle': {
    'description': '横滚角对比（带传感器噪声）',
    'robot_expressions': {...},
    'noise_config': {
        'default': {'mean': 0.0, 'std': 0.017, 'seed': 42},
    },
}
```

### 示例4: 位置噪声（单位：米）

对于位置数据，噪声通常以米为单位：

```python
'x_position': {
    'description': 'X位置对比（带测量噪声）',
    'robot_expressions': {...},
    'noise_config': {
        'default': {'mean': 0.0, 'std': 0.001, 'seed': 42},  # 1mm噪声
        'methods': {
            'React-PPO': {'mean': 0.0, 'std': 0.002, 'seed': 42},  # 2mm噪声
        },
    },
}
```

## 注意事项

1. **噪声是在表达式计算后添加的**：噪声会添加到计算后的数据上，而不是原始数据。
2. **噪声是独立的**：每次调用都会生成新的随机噪声（除非设置了seed）。
3. **噪声不影响原始数据**：噪声只影响绘图，不会修改原始DataFrame。
4. **std=0时不添加噪声**：如果 `std <= 0.0`，则不会添加任何噪声。
5. **seed用于可重复性**：设置相同的seed可以确保每次运行产生相同的噪声序列。

## 在代码中修改

在 `generate_comparison_from_single_data.py` 中，找到 `PLOT_CONFIGURATIONS` 字典，为需要添加噪声的图表添加 `noise_config` 字段即可。

例如，为 `roll_angle` 图表添加噪声：

```python
'roll_angle': {
    'description': '各方法机器狗横滚角和平台横滚角随时间变化曲线对比',
    'robot_expressions': {...},
    'noise_config': {
        'default': {'mean': 0.0, 'std': 0.01, 'seed': 42},
    },
}
```

