#!/usr/bin/env python3
"""
性能对比测试脚本
测试原有代码 vs 优化后代码的性能差异
"""

import time
import numpy as np
import sys
import os

# 添加路径
sys.path.append('/home/user/IsaacLab/source/isaaclab/isaaclab/envs/mdp')

try:
    from vessels import VesselControlSystem
    print("✓ 成功导入VesselControlSystem")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

def test_original_approach(num_envs=1024, steps=100):
    """测试原有方法：每个环境独立的平台系统"""
    print(f"\n=== 测试原有方法：{num_envs}个环境，{steps}步 ===")
    
    # 创建多个独立的平台系统
    start_time = time.time()
    systems = {}
    for i in range(num_envs):
        systems[i] = VesselControlSystem(
            target_position=[10, 10, 0.8 * np.pi],
            initial_eta=np.array([0, 0, 0, 0, 0, 0]),
            initial_nu=np.array([0, 0, 0, 0, 0, 0]),
            dt=0.02
        )
    creation_time = time.time() - start_time
    print(f"创建{num_envs}个独立系统耗时: {creation_time:.4f}秒")
    
    # 模拟每个环境独立计算
    start_time = time.time()
    for step in range(steps):
        for i in range(num_envs):
            # 模拟不同的位置和速度
            pose = np.random.randn(6) * 0.1
            nu = np.random.randn(6) * 0.1
            time_val = step * 0.02
            
            # 计算加速度
            acc, eta_dot = systems[i].step(pose, nu, time_val)
    
    computation_time = time.time() - start_time
    total_time = creation_time + computation_time
    
    print(f"计算{steps}步耗时: {computation_time:.4f}秒")
    print(f"总耗时: {total_time:.4f}秒")
    print(f"平均每步耗时: {computation_time/steps:.6f}秒")
    print(f"平均每环境每步耗时: {computation_time/(steps*num_envs):.8f}秒")
    
    return total_time, computation_time

def test_optimized_approach(num_envs=1024, steps=100):
    """测试优化方法：所有环境共用一个平台系统"""
    print(f"\n=== 测试优化方法：{num_envs}个环境，{steps}步 ===")
    
    # 创建一个共享的平台系统
    start_time = time.time()
    shared_system = VesselControlSystem(
        target_position=[10, 10, 0.8 * np.pi],
        initial_eta=np.array([0, 0, 0, 0, 0, 0]),
        initial_nu=np.array([0, 0, 0, 0, 0, 0]),
        dt=0.02
    )
    creation_time = time.time() - start_time
    print(f"创建1个共享系统耗时: {creation_time:.4f}秒")
    
    # 模拟批量计算
    start_time = time.time()
    for step in range(steps):
        # 批量生成所有环境的位置和速度
        pose_batch = np.random.randn(num_envs, 6) * 0.1
        nu_batch = np.random.randn(num_envs, 6) * 0.1
        time_val = step * 0.02
        
        # 批量计算所有环境的加速度
        acc_batch = np.zeros((num_envs, 6))
        eta_dot_batch = np.zeros((num_envs, 6))
        
        for i in range(num_envs):
            # 更新共享系统的状态
            shared_system.eta[:] = pose_batch[i]
            shared_system.nu[:] = nu_batch[i]
            
            # 计算当前环境的加速度
            acc, eta_dot = shared_system.step(pose_batch[i], nu_batch[i], time_val)
            acc_batch[i] = acc
            eta_dot_batch[i] = eta_dot
    
    computation_time = time.time() - start_time
    total_time = creation_time + computation_time
    
    print(f"计算{steps}步耗时: {computation_time:.4f}秒")
    print(f"总耗时: {total_time:.4f}秒")
    print(f"平均每步耗时: {computation_time/steps:.6f}秒")
    print(f"平均每环境每步耗时: {computation_time/(steps*num_envs):.8f}秒")
    
    return total_time, computation_time

def main():
    """主测试函数"""
    print("=== 船舶控制系统性能对比测试 ===")
    
    # 测试参数
    test_cases = [
        (100, 50),    # 小规模测试
        (500, 100),   # 中等规模测试
        (1024, 200),  # 大规模测试
    ]
    
    results = []
    
    for num_envs, steps in test_cases:
        print(f"\n{'='*60}")
        print(f"测试规模: {num_envs}个环境, {steps}步")
        print(f"{'='*60}")
        
        # 测试原有方法
        original_total, original_comp = test_original_approach(num_envs, steps)
        
        # 测试优化方法
        optimized_total, optimized_comp = test_optimized_approach(num_envs, steps)
        
        # 计算性能提升
        speedup_total = original_total / optimized_total
        speedup_comp = original_comp / optimized_comp
        
        print(f"\n--- 性能对比结果 ---")
        print(f"总时间提升: {speedup_total:.2f}x")
        print(f"计算时间提升: {speedup_comp:.2f}x")
        print(f"时间节省: {((original_total - optimized_total) / original_total * 100):.1f}%")
        
        results.append({
            'num_envs': num_envs,
            'steps': steps,
            'original_total': original_total,
            'optimized_total': optimized_total,
            'speedup': speedup_total,
            'time_saved': ((original_total - optimized_total) / original_total * 100)
        })
    
    # 总结报告
    print(f"\n{'='*80}")
    print("性能测试总结报告")
    print(f"{'='*80}")
    
    for result in results:
        print(f"环境数: {result['num_envs']:4d}, 步数: {result['steps']:3d} | "
              f"提升: {result['speedup']:5.2f}x | 节省: {result['time_saved']:5.1f}%")
    
    # 计算平均提升
    avg_speedup = np.mean([r['speedup'] for r in results])
    avg_saved = np.mean([r['time_saved'] for r in results])
    
    print(f"\n平均性能提升: {avg_speedup:.2f}x")
    print(f"平均时间节省: {avg_saved:.1f}%")
    
    print(f"\n优化效果分析:")
    print(f"1. 内存使用: 从{test_cases[-1][0]}个系统减少到1个系统")
    print(f"2. 初始化时间: 大幅减少（只创建1个系统）")
    print(f"3. 计算效率: 提升{avg_speedup:.1f}倍")
    print(f"4. 适用场景: 多环境并行训练")

if __name__ == "__main__":
    main()
