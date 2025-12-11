# 对比实验脚本使用指南

## 一、如何修改不同方法的目录

### 方法1：通过命令行参数（推荐）

在运行脚本时，使用 `--data_dirs` 参数指定各方法的数据目录：

```bash
python scripts/tools/compare_methods.py \
    --data_dirs \
        "React-PPO:/home/user/IsaacLab/training_data/react_ppo" \
        "React-MPC:/home/user/IsaacLab/training_data/react_mpc" \
        "Oracle-PPO:/home/user/IsaacLab/training_data/oracle_ppo" \
        "Ours w/o Prediction:/home/user/IsaacLab/training_data/ours_wo_prediction" \
        "Ours:/home/user/IsaacLab/training_data/ours" \
    --output_dir /home/user/IsaacLab/comparison_results
```

**格式说明**：
- 格式：`"方法名:数据目录路径"`
- 多个方法用空格或换行分隔
- 方法名可以自定义，但建议使用预定义的方法名（React-PPO, React-MPC, Oracle-PPO, Ours w/o Prediction, Ours）

### 方法2：修改代码中的默认配置

如果需要经常使用相同的目录配置，可以修改 `compare_methods.py` 文件，在 `main()` 函数中添加默认配置：

```python
# 在 main() 函数开始处添加
if not args.data_dirs:
    args.data_dirs = [
        "React-PPO:/home/user/IsaacLab/training_data/react_ppo",
        "React-MPC:/home/user/IsaacLab/training_data/react_mpc",
        "Oracle-PPO:/home/user/IsaacLab/training_data/oracle_ppo",
        "Ours w/o Prediction:/home/user/IsaacLab/training_data/ours_wo_prediction",
        "Ours:/home/user/IsaacLab/training_data/ours",
    ]
```

## 二、如何对读取的数据进行加减乘除操作

### 方法1：通过命令行参数（推荐）

使用 `--data_transform` 参数指定数据变换：

```bash
python scripts/tools/compare_methods.py \
    --data_dirs \
        "Ours:/home/user/IsaacLab/training_data" \
    --data_transform '{"Ours": {"control_error": {"operation": "multiply", "value": 1000}}}' \
    --output_dir /home/user/IsaacLab/comparison_results
```

**JSON格式说明**：
```json
{
    "方法名": {
        "列名": {
            "operation": "add/subtract/multiply/divide",
            "value": 数值
        }
    }
}
```

**示例**：

1. **将控制误差乘以1000**（转换为毫弧度）：
```bash
--data_transform '{"Ours": {"control_error": {"operation": "multiply", "value": 1000}}}'
```

2. **将多个列进行变换**：
```bash
--data_transform '{
    "Ours": {
        "control_error": {"operation": "multiply", "value": 1000},
        "prediction_error": {"operation": "multiply", "value": 1000},
        "time": {"operation": "divide", "value": 60}
    }
}'
```

3. **对不同方法应用不同变换**：
```bash
--data_transform '{
    "React-PPO": {
        "control_error": {"operation": "multiply", "value": 1000}
    },
    "Ours": {
        "control_error": {"operation": "multiply", "value": 1000},
        "energy_consumption": {"operation": "divide", "value": 1000}
    }
}'
```

### 方法2：修改代码中的 `load_csv_data` 函数

如果需要更复杂的数据处理，可以直接修改 `load_csv_data` 函数：

```python
def load_csv_data(data_dir: str, data_transform: dict = None):
    # ... 现有代码 ...
    
    # 在返回之前添加自定义处理
    # 例如：将角度从弧度转换为度
    if 'control_error' in combined_df.columns:
        combined_df['control_error_deg'] = combined_df['control_error'] * 180 / np.pi
    
    # 例如：计算累积误差
    if 'control_error' in combined_df.columns:
        combined_df['cumulative_error'] = combined_df['control_error'].cumsum()
    
    return combined_df
```

### 方法3：在数据加载后进行处理

在 `main()` 函数中，数据加载后可以添加处理逻辑：

```python
# 在 main() 函数中，数据加载后
for method_name, df in method_data.items():
    if df is not None:
        # 示例1：将控制误差转换为度
        if 'control_error' in df.columns:
            df['control_error_deg'] = df['control_error'] * 180 / np.pi
        
        # 示例2：添加偏移量
        if 'robot_roll' in df.columns:
            df['robot_roll'] = df['robot_roll'] + 0.1  # 添加0.1弧度偏移
        
        # 示例3：归一化
        if 'energy_consumption' in df.columns:
            max_energy = df['energy_consumption'].max()
            if max_energy > 0:
                df['energy_normalized'] = df['energy_consumption'] / max_energy
```

## 三、常用数据变换示例

### 1. 单位转换

**弧度转度**：
```bash
--data_transform '{"Ours": {"control_error": {"operation": "multiply", "value": 57.2958}}}'
```

**秒转分钟**：
```bash
--data_transform '{"Ours": {"time": {"operation": "divide", "value": 60}}}'
```

### 2. 数据缩放

**放大1000倍**（用于显示小数值）：
```bash
--data_transform '{"Ours": {"prediction_error": {"operation": "multiply", "value": 1000}}}'
```

**缩小1000倍**：
```bash
--data_transform '{"Ours": {"energy_consumption": {"operation": "divide", "value": 1000}}}'
```

### 3. 数据偏移

**添加偏移量**：
```bash
--data_transform '{"Ours": {"robot_roll": {"operation": "add", "value": 0.1}}}'
```

**减去偏移量**：
```bash
--data_transform '{"Ours": {"platform_roll": {"operation": "subtract", "value": 0.05}}}'
```

## 四、完整使用示例

### 示例1：基本对比（无数据变换）

```bash
python scripts/tools/compare_methods.py \
    --data_dirs \
        "React-PPO:/path/to/react_ppo" \
        "Ours:/path/to/ours" \
    --output_dir /home/user/IsaacLab/comparison_results
```

### 示例2：带数据变换的对比

```bash
python scripts/tools/compare_methods.py \
    --data_dirs \
        "React-PPO:/path/to/react_ppo" \
        "Ours:/path/to/ours" \
    --data_transform '{
        "React-PPO": {
            "control_error": {"operation": "multiply", "value": 1000}
        },
        "Ours": {
            "control_error": {"operation": "multiply", "value": 1000},
            "prediction_error": {"operation": "multiply", "value": 1000},
            "time": {"operation": "divide", "value": 60}
        }
    }' \
    --output_dir /home/user/IsaacLab/comparison_results
```

### 示例3：使用配置文件（推荐用于复杂变换）

创建一个JSON配置文件 `data_transform.json`：

```json
{
    "React-PPO": {
        "control_error": {"operation": "multiply", "value": 1000}
    },
    "React-MPC": {
        "control_error": {"operation": "multiply", "value": 1000}
    },
    "Oracle-PPO": {
        "control_error": {"operation": "multiply", "value": 1000}
    },
    "Ours w/o Prediction": {
        "control_error": {"operation": "multiply", "value": 1000}
    },
    "Ours": {
        "control_error": {"operation": "multiply", "value": 1000},
        "prediction_error": {"operation": "multiply", "value": 1000}
    }
}
```

然后在脚本中读取：

```bash
# 读取配置文件
TRANSFORM_CONFIG=$(cat data_transform.json)

python scripts/tools/compare_methods.py \
    --data_dirs \
        "React-PPO:/path/to/react_ppo" \
        "Ours:/path/to/ours" \
    --data_transform "$TRANSFORM_CONFIG" \
    --output_dir /home/user/IsaacLab/comparison_results
```

## 五、注意事项

1. **数据变换顺序**：变换在数据加载后、排序前应用
2. **列名匹配**：确保列名完全匹配（区分大小写）
3. **数值类型**：确保变换后的数据类型正确
4. **除零检查**：除法操作会自动检查除零情况
5. **JSON格式**：命令行中的JSON字符串需要用单引号包裹，内部使用双引号

## 六、调试技巧

如果数据变换没有生效，可以：

1. **检查列名**：打印DataFrame的列名
   ```python
   print(df.columns.tolist())
   ```

2. **检查变换配置**：脚本会打印应用的变换信息

3. **验证数据**：在变换后检查数据范围
   ```python
   print(df['control_error'].describe())
   ```

## 七、扩展功能

如果需要更复杂的数据处理（如滤波、插值等），可以：

1. 在 `load_csv_data` 函数中添加处理逻辑
2. 创建新的数据处理函数
3. 在数据加载后调用处理函数

例如，添加移动平均滤波：

```python
def apply_moving_average(df, column_name, window_size=10):
    """应用移动平均滤波"""
    if column_name in df.columns:
        df[f'{column_name}_smooth'] = df[column_name].rolling(window=window_size).mean()
    return df
```

然后在 `main()` 函数中调用：

```python
for method_name, df in method_data.items():
    if df is not None:
        df = apply_moving_average(df, 'control_error', window_size=10)
        method_data[method_name] = df
```

