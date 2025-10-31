#!/usr/bin/env python3
"""
测试共享控制优化效果
"""

import numpy as np
import time
import sys
import os

# 添加路径以导入vessels模块
sys.path.append('/home/user/IsaacLab/source/isaaclab/isaaclab/envs/mdp')
from vessels import VesselControlSystem

def test_shared_vs_individual_control():
    """测试共享控制 vs 独立控制的性能差异"""
    
    print("=== 共享控制 vs 独立控制性能对比 ===")
    
    # 测试参数
    target_position = [10, 10, 0.8 * np.pi]
    initial_eta = np.array([0, 0, 0, 0, 0, 0], dtype=float)
    initial_nu = np.array([0, 0, 0, 0, 0, 0], dtype=float)
    dt = 0.01
    
    # 测试不同环境数量
    test_cases = [1, 10, 50, 100, 500, 1000, 1024]
    
    for num_envs in test_cases:
        print(f"\n--- 测试 {num_envs} 个环境 ---")
        
        # 生成随机状态数据
        poses = np.random.randn(num_envs, 6) * 0.1
        nus = np.random.randn(num_envs, 6) * 0.1
        current_time = 1.0
        
        # 方法1：独立控制（每个环境都有自己的系统）
        print("方法1：独立控制（每个环境独立计算）")
        start_time = time.time()
        
        individual_systems = []
        for i in range(num_envs):
            system = VesselControlSystem(
                target_position=target_position,
                initial_eta=poses[i],
                initial_nu=nus[i],
                dt=dt
            )
            individual_systems.append(system)
        
        for i in range(num_envs):
            acc, eta_dot = individual_systems[i].step(poses[i], nus[i], current_time)
        
        method1_time = time.time() - start_time
        print(f"  独立控制耗时: {method1_time*1000:.2f}ms")
        print(f"  单次step平均: {method1_time*1000/num_envs:.3f}ms")
        
        # 方法2：共享控制（所有环境使用同一个系统）
        print("方法2：共享控制（所有环境同步运动）")
        start_time = time.time()
        
        # 创建共享系统
        shared_system = VesselControlSystem(
            target_position=target_position,
            initial_eta=poses[0],
            initial_nu=nus[0],
            dt=dt
        )
        
        # 只计算一次控制指令
        shared_system.eta[:] = poses[0]
        shared_system.nu[:] = nus[0]
        acc, eta_dot = shared_system.step(poses[0], nus[0], current_time)
        
        # 将相同指令应用到所有环境
        for i in range(num_envs):
            # 这里只是模拟，实际中所有环境会得到相同的acc和eta_dot
            pass
        
        method2_time = time.time() - start_time
        print(f"  共享控制耗时: {method2_time*1000:.2f}ms")
        print(f"  单次step平均: {method2_time*1000/num_envs:.3f}ms")
        
        # 性能提升计算
        if method1_time > 0:
            speedup = method1_time / method2_time
            print(f"  性能提升: {speedup:.1f}x")
            
            # 预计1024环境的性能
            if num_envs >= 100:
                estimated_1024_individual = method1_time * 1024 / num_envs
                estimated_1024_shared = method2_time * 1024 / num_envs
                print(f"  预计1024环境 - 独立控制: {estimated_1024_individual:.2f}秒")
                print(f"  预计1024环境 - 共享控制: {estimated_1024_shared:.2f}秒")
                print(f"  预计1024环境性能提升: {estimated_1024_individual/estimated_1024_shared:.1f}x")

def test_control_consistency():
    """测试共享控制的运动一致性"""
    
    print("\n=== 共享控制运动一致性测试 ===")
    
    # 创建共享系统
    target_position = [10, 10, 0.8 * np.pi]
    initial_eta = np.array([0, 0, 0, 0, 0, 0], dtype=float)
    initial_nu = np.array([0, 0, 0, 0, 0, 0], dtype=float)
    dt = 0.01
    
    shared_system = VesselControlSystem(
        target_position=target_position,
        initial_eta=initial_eta,
        initial_nu=initial_nu,
        dt=dt
    )
    
    # 模拟多个环境的状态（稍微不同）
    num_envs = 5
    poses = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1, 0.0, 0.0, 0.0]
    ])
    nus = np.zeros((num_envs, 6))
    current_time = 1.0
    
    print("各环境初始状态:")
    for i in range(num_envs):
        print(f"  环境{i}: 位置={poses[i][:3]}, 角度={poses[i][3:]}")
    
    # 使用共享控制计算控制指令
    shared_system.eta[:] = poses[0]  # 使用第一个环境的状态
    shared_system.nu[:] = nus[0]
    acc, eta_dot = shared_system.step(poses[0], nus[0], current_time)
    
    print(f"\n共享控制指令:")
    print(f"  加速度: {acc}")
    print(f"  位置导数: {eta_dot}")
    
    print(f"\n所有{num_envs}个环境将获得相同的控制指令")
    print("这意味着所有平台将执行同步运动，就像'同步舞蹈'一样")

if __name__ == "__main__":
    test_shared_vs_individual_control()
    test_control_consistency()
