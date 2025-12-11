# 维度不匹配错误修复总结

## 🐛 错误信息

```
IndexError: index 2 is out of bounds for dimension 2 with size 2
File: platform_predictor.py, line 1584
weights[:, :, 2] = 1.5  # roll_ang_vel权重
```

## 🔍 问题原因

在将预测器从6自由度改为2维（只预测roll和pitch）后，训练代码中仍然有多个地方试图访问索引2和3（roll_ang_vel和pitch_ang_vel），导致维度不匹配错误。

## ✅ 修复内容

### 1. 训练损失权重设置
**文件**: `platform_predictor.py`, line 1580-1584
- **修复前**: 试图设置索引2和3的权重（roll_ang_vel和pitch_ang_vel）
- **修复后**: 只设置索引0和1的权重（roll和pitch）

### 2. 目标数据归一化
**文件**: `platform_predictor.py`, line 2050-2057
- **修复前**: 试图归一化索引2和3（roll_ang_vel和pitch_ang_vel）
- **修复后**: 只归一化索引0和1（roll和pitch）

### 3. 上下文内目标数据归一化
**文件**: `platform_predictor.py`, line 2091-2095
- **修复前**: 试图归一化索引2和3
- **修复后**: 只归一化索引0和1

### 4. 反归一化代码（3处）
**文件**: `platform_predictor.py`
- **位置1**: `predict_future_from_observations`方法（line ~800）
- **位置2**: `predict_current_from_delayed_history`方法（line ~970）
- **位置3**: `predict_future`方法（line ~1149）
- **修复前**: 试图反归一化索引2和3，并提取角速度
- **修复后**: 只反归一化索引0和1，角速度设为0

### 5. 趋势损失计算
**文件**: `platform_predictor.py`, line 1599-1604
- **修复前**: 计算角速度变化趋势（索引2:4）
- **修复后**: 计算姿态变化趋势（索引0:2）

### 6. 评估代码
**文件**: `platform_predictor.py`, line 2691-2695
- **修复前**: 试图提取索引2和3的角速度
- **修复后**: 只提取索引0和1的姿态，角速度设为0

### 7. 监控代码
**文件**: `platform_predictor.py`, line 2462-2466
- **修复前**: 试图反归一化索引2和3用于监控
- **修复后**: 只反归一化索引0和1

### 8. 另一个训练方法的权重设置
**文件**: `platform_predictor.py`, line 2214-2219
- **修复前**: 试图设置索引2和3的权重
- **修复后**: 只设置索引0和1的权重

## 📊 修复统计

- **修复的方法数**: 8个位置
- **修复的代码行数**: 约20行
- **主要修改**: 移除对索引2和3的访问，角速度设为0

## ✅ 验证

- [x] 所有索引访问已修复
- [x] 代码通过lint检查
- [x] 维度匹配：输出2维（roll, pitch）

## 🎯 关键改进

1. **维度一致性**: 所有代码现在都使用2维输出（roll, pitch）
2. **角速度处理**: 角速度暂时设为0，保持接口兼容性
3. **代码清晰**: 添加了注释说明维度变化

所有修复已完成，代码现在应该可以正常运行了。

