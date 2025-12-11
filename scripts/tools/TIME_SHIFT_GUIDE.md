# 时间相位调整指南

本指南说明如何在绘图时调整各方法的时间相位（延迟或快进）。

## 功能说明

在绘图时，可以为每个方法单独设置时间偏移，用于：
- 调整方法的相位，观察延迟或提前的效果
- 对齐不同方法的响应时间
- 分析方法的响应延迟

## 时间偏移配置格式

时间偏移使用字典格式，支持以下配置：

```python
'time_shift_config': {
    'default': 0.0,  # 全局默认时间偏移（秒），0表示不偏移
    'methods': {  # 方法特定时间偏移（会覆盖默认值）
        'Ours': 2.0,      # 延迟2秒（向右移动）
        'React-MPC': -1.0, # 快进1秒（向左移动）
    },
}
```

## 参数说明

### default (全局默认偏移)
- **类型**: float
- **默认值**: 0.0
- **说明**: 所有方法的默认时间偏移。如果某个方法没有在 `methods` 中指定，则使用此值。

### methods (方法特定偏移)
- **类型**: dict
- **说明**: 为不同方法设置不同的时间偏移。键为方法名，值为偏移量（秒）。

### 偏移量含义
- **正数**：延迟（向右移动），例如 `2.0` 表示延迟2秒
- **负数**：快进（向左移动），例如 `-1.0` 表示快进1秒
- **0.0**：不偏移

## 配置方式

### 方式1: 全局默认时间偏移

在图表配置中添加 `time_shift_config`，设置全局默认偏移：

```python
'roll_angle': {
    'description': '横滚角对比',
    'robot_expressions': {...},
    'time_shift_config': {
        'default': 1.0,  # 所有方法延迟1秒
    },
}
```

### 方式2: 方法特定时间偏移

为不同方法设置不同的时间偏移：

```python
'roll_angle': {
    'description': '横滚角对比',
    'robot_expressions': {...},
    'time_shift_config': {
        'default': 0.0,  # 默认不偏移
        'methods': {  # 方法特定偏移
            'Ours': 2.0,      # Ours方法延迟2秒
            'React-MPC': -1.0, # React-MPC方法快进1秒
            'Oracle-PPO': 0.5, # Oracle-PPO方法延迟0.5秒
        },
    },
}
```

## 使用示例

### 示例1: 让Ours方法延迟2秒

```python
'roll_angle': {
    'description': '横滚角对比',
    'robot_expressions': {...},
    'time_shift_config': {
        'default': 0.0,
        'methods': {
            'Ours': 2.0,  # Ours方法延迟2秒
        },
    },
}
```

### 示例2: 让React-MPC快进1秒

```python
'pitch_angle': {
    'description': '俯仰角对比',
    'robot_expressions': {...},
    'time_shift_config': {
        'default': 0.0,
        'methods': {
            'React-MPC': -1.0,  # React-MPC方法快进1秒
        },
    },
}
```

### 示例3: 同时调整多个方法的相位

```python
'roll_angle': {
    'description': '横滚角对比（相位调整）',
    'robot_expressions': {...},
    'time_shift_config': {
        'default': 0.0,
        'methods': {
            'Ours': 2.0,        # 延迟2秒
            'Oracle-PPO': 1.0,  # 延迟1秒
            'React-MPC': -0.5,  # 快进0.5秒
        },
    },
}
```

### 示例4: 对齐响应时间

假设Ours方法有2秒的响应延迟，可以通过负偏移来对齐：

```python
'roll_angle': {
    'description': '横滚角对比（对齐响应时间）',
    'robot_expressions': {...},
    'time_shift_config': {
        'default': 0.0,
        'methods': {
            'Ours': -2.0,  # 快进2秒，对齐响应时间
        },
    },
}
```

## 注意事项

1. **时间偏移只影响绘图**：时间偏移只影响图表中的显示，不会修改原始数据。
2. **时间偏移是累加的**：如果同时设置了全局偏移和方法特定偏移，方法特定偏移会覆盖全局偏移。
3. **时间范围过滤**：时间偏移在时间范围过滤之后应用，所以偏移后的时间可能超出原始数据范围。
4. **单位是秒**：所有时间偏移的单位都是秒（seconds），与 `time` 列的单位一致。
5. **正负值含义**：
   - 正数：延迟（曲线向右移动）
   - 负数：快进（曲线向左移动）

## 在代码中修改

在 `generate_comparison_from_single_data.py` 中，找到 `PLOT_CONFIGURATIONS` 字典，为需要调整相位的图表添加 `time_shift_config` 字段即可。

例如，为 `roll_angle` 图表添加时间偏移：

```python
'roll_angle': {
    'description': '各方法机器狗横滚角和平台横滚角随时间变化曲线对比',
    'robot_expressions': {...},
    'time_shift_config': {
        'default': 0.0,
        'methods': {
            'Ours': 2.0,      # 延迟2秒
            'React-MPC': -1.0, # 快进1秒
        },
    },
}
```

## 应用场景

1. **分析响应延迟**：通过调整相位，观察不同方法的响应延迟差异
2. **对齐峰值**：将不同方法的峰值对齐，便于比较
3. **相位对比**：观察不同方法相对于平台运动的相位关系
4. **延迟补偿**：如果某个方法有已知的延迟，可以通过负偏移来补偿

