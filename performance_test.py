#!/usr/bin/env python3
"""
性能测试脚本 - 验证船舶控制系统优化效果
"""

import time
import numpy as np
import sys
import os

# 添加路径
sys.path.append('/home/user/IsaacLab/source/isaaclab/isaaclab/envs/mdp')

def test_vessel_control_performance():
    """测试船舶控制系统性能"""
    from vessels import VesselControlSystem
    
    print("=== 船舶控制系统性能测试 ===")
    
    # 测试参数
    num_environments = 100  # 测试100个环境
    num_steps = 1000       # 每个环境运行1000步
    
    print(f"测试环境数量: {num_environments}")
    print(f"每环境步数: {num_steps}")
    
    # 记录开始时间
    start_time = time.time()
    
    # 创建多个船舶控制系统实例
    vessel_systems = []
    for i in range(num_environments):
        # 每个环境有不同的初始状态（模拟随机性）
        initial_eta = np.array([0, 0, 0, 0, 0, 0]) + np.random.normal(0, 0.1, 6)
        initial_nu = np.random.normal(0, 0.1, 6)
        
        vessel_system = VesselControlSystem(
            target_position=[10, 10, np.pi],
            initial_eta=initial_eta,
            initial_nu=initial_nu,
            dt=0.02
        )
        vessel_systems.append(vessel_system)
    
    creation_time = time.time()
    print(f"创建{num_environments}个船舶系统耗时: {creation_time - start_time:.2f}秒")
    
    # 运行仿真
    simulation_start = time.time()
    
    for step in range(num_steps):
        for i, vessel_system in enumerate(vessel_systems):
            # 模拟当前状态
            current_eta = vessel_system.eta + np.random.normal(0, 0.01, 6)
            current_nu = vessel_system.nu + np.random.normal(0, 0.01, 6)
            current_time = step * 0.02
            
            # 执行一步计算
            nu_dot, eta_dot = vessel_system.step(current_eta, current_nu, current_time)
            
            # 更新状态
            vessel_system.eta = current_eta + eta_dot * 0.02
            vessel_system.nu = current_nu + nu_dot * 0.02
        
        # 每100步显示进度
        if (step + 1) % 100 == 0:
            elapsed = time.time() - simulation_start
            print(f"完成 {step + 1}/{num_steps} 步, 耗时: {elapsed:.2f}秒")
    
    total_time = time.time() - start_time
    simulation_time = time.time() - simulation_start
    
    print(f"\n=== 性能测试结果 ===")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"仿真耗时: {simulation_time:.2f}秒")
    print(f"创建系统耗时: {creation_time - start_time:.2f}秒")
    print(f"平均每步耗时: {simulation_time / (num_steps * num_environments) * 1000:.2f}毫秒")
    print(f"每秒可处理步数: {num_steps * num_environments / simulation_time:.0f}")

def test_memory_usage():
    """测试内存使用情况"""
    import psutil
    import gc
    from vessels import VesselControlSystem
    
    print("\n=== 内存使用测试 ===")
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"初始内存使用: {initial_memory:.1f} MB")
    
    # 创建大量船舶系统
    vessel_systems = []
    for i in range(50):
        vessel_system = VesselControlSystem(
            target_position=[10, 10, np.pi],
            initial_eta=np.random.normal(0, 0.1, 6),
            initial_nu=np.random.normal(0, 0.1, 6),
            dt=0.02
        )
        vessel_systems.append(vessel_system)
    
    after_creation_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"创建50个系统后内存使用: {after_creation_memory:.1f} MB")
    print(f"每个系统平均内存: {(after_creation_memory - initial_memory) / 50:.1f} MB")
    
    # 清理
    del vessel_systems
    gc.collect()
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"清理后内存使用: {final_memory:.1f} MB")

if __name__ == "__main__":
    try:
        test_vessel_control_performance()
        test_memory_usage()
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
