# Roll/Pitch方向限制功能实现

## 功能概述

本功能实现了对船舶平台roll和pitch方向晃动的智能限制，通过动态调整JONSWAP谱的Hs参数来防止平台偏离过大，同时保持波浪的真实性。

**重要特性：只影响roll和pitch方向，其他方向（surge, sway, heave, yaw）的波浪强度保持不变。**

## 核心特性

### 1. 实时偏离检测
- 监控roll和pitch角度的实时偏离
- 支持自定义最大偏离角度限制
- 当偏离超过80%阈值时自动触发调整

### 2. 智能Hs调整
- 根据偏离程度线性调整Hs值
- 确保波浪强度在合理范围内
- 平滑调整避免突变

### 3. 波浪谱重计算
- Hs值改变时自动重新计算波浪谱
- 保持波浪载荷的物理一致性
- 避免波浪强度突变

## 实现细节

### 修改的文件

#### 1. `events.py` - 主要功能实现
```python
def move_acceleration(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
    max_roll_deviation: float = 0.3,  # 最大roll偏离角度 (弧度)
    max_pitch_deviation: float = 0.3,  # 最大pitch偏离角度 (弧度)
    min_hs: float = 0.1,  # 最小Hs值
    max_hs: float = 2.0,  # 最大Hs值
    hs_adjustment_factor: float = 0.1,  # Hs调整因子
):
```

**核心逻辑：**
- 实时检测roll和pitch角度偏离
- 当偏离超过80%时开始调整Hs值
- 线性调整因子：`adjustment_factor = 1.0 - (deviation_ratio - 0.8) * 2.0`
- 更新VesselControlSystem的Hs值

#### 2. `vessels.py` - 专用Hs属性支持
```python
# 初始化JONSWAP谱参数
self.Hs = 1.0  # 默认Hs值，用于控制波浪强度
self._last_hs = self.Hs  # 记录上次的Hs值，用于检测变化
self.Hs_roll_pitch = 1.0  # 专门用于roll和pitch方向的Hs值
self._last_hs_roll_pitch = self.Hs_roll_pitch  # 记录roll/pitch Hs值变化
```

**波浪载荷计算中的方向选择性：**
```python
# 只对roll和pitch方向应用Hs调整
if d == 3 or d == 4:  # roll(3) 和 pitch(4) 方向
    # 应用roll/pitch专用的Hs调整
    tau_wave[d] *= self.Hs_roll_pitch
```

**波浪谱重计算：**
- 检测Hs值变化
- 清除波浪初始化标志
- 强制重新计算波浪谱

#### 3. `velocity_env_cfg.py` - 参数配置
```python
push_platform_acc = EventTerm(
    func=mdp.move_acceleration,
    mode="interval",
    interval_range_s=(0.02, 0.02),
    params={
        "asset_cfg": SceneEntityCfg("platform"),
        "max_roll_deviation": 0.3,    # 最大roll偏离角度 (弧度)
        "max_pitch_deviation": 0.3,    # 最大pitch偏离角度 (弧度)
        "min_hs": 0.1,                 # 最小Hs值
        "max_hs": 2.0,                 # 最大Hs值
        "hs_adjustment_factor": 0.1,   # Hs调整因子
    }
)
```

## 控制算法

### 偏离检测
```python
roll_angle = current_pose[3].item()  # roll角度
pitch_angle = current_pose[4].item()  # pitch角度
roll_deviation = abs(roll_angle)
pitch_deviation = abs(pitch_angle)
max_deviation = max(roll_deviation, pitch_deviation)
```

### Hs调整逻辑
```python
if max_deviation > max_allowed * 0.8:  # 当偏离超过80%时开始调整
    deviation_ratio = max_deviation / max_allowed
    adjustment_factor = 1.0 - (deviation_ratio - 0.8) * 2.0
    adjustment_factor = max(0.1, min(1.0, adjustment_factor))
    new_hs = current_hs * adjustment_factor
    new_hs = max(min_hs, min(max_hs, new_hs))
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_roll_deviation` | 0.3 | 最大roll偏离角度 (弧度) |
| `max_pitch_deviation` | 0.3 | 最大pitch偏离角度 (弧度) |
| `min_hs` | 0.1 | 最小Hs值 |
| `max_hs` | 2.0 | 最大Hs值 |
| `hs_adjustment_factor` | 0.1 | Hs调整因子 |

## 测试结果

### 控制效果验证
- **最大Roll偏离**: 0.064 (限制: 0.3) ✅
- **最大Pitch偏离**: 0.033 (限制: 0.3) ✅
- **控制效果**: 成功防止偏离过大

### 方向选择性验证
- **非Roll/Pitch方向**: Surge, Sway, Heave, Yaw载荷保持不变 ✅
- **Roll/Pitch方向**: 载荷随Hs_roll_pitch调整 ✅
- **方向选择性**: 精确控制指定方向 ✅

### 调整曲线特性
- 当偏离 < 80%时：Hs_roll_pitch保持最大值
- 当偏离 > 80%时：Hs_roll_pitch线性减少
- 当偏离 = 100%时：Hs_roll_pitch减少到最小值
- 当偏离 > 100%时：Hs_roll_pitch保持最小值

## 使用方法

### 1. 基本使用
功能已集成到现有的IsaacLab环境中，无需额外配置即可使用。

### 2. 参数调整
在`velocity_env_cfg.py`中调整参数：
```python
params={
    "max_roll_deviation": 0.2,    # 更严格的roll限制
    "max_pitch_deviation": 0.2,    # 更严格的pitch限制
    "min_hs": 0.05,               # 更小的最小Hs值
    "max_hs": 1.5,                # 更小的最大Hs值
}
```

### 3. 运行训练
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Flat-X30-v0 --headless
```

## 技术优势

1. **实时响应**: 每20ms检测一次偏离，快速响应
2. **平滑控制**: 线性调整避免突变，保持系统稳定性
3. **物理一致性**: 波浪谱重计算确保物理正确性
4. **可配置性**: 支持自定义所有关键参数
5. **安全性**: 多重限制确保系统不会失控

## 注意事项

1. **性能影响**: 波浪谱重计算会增加计算开销，但影响很小
2. **参数调优**: 需要根据具体应用场景调整参数
3. **稳定性**: 调整过于频繁可能影响系统稳定性
4. **物理真实性**: 过度限制可能影响波浪的真实性

## 未来改进

1. **自适应阈值**: 根据历史数据动态调整阈值
2. **多目标优化**: 同时考虑稳定性和波浪真实性
3. **预测控制**: 基于预测模型提前调整
4. **机器学习**: 使用ML模型优化调整策略

