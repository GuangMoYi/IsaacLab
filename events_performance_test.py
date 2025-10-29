#!/usr/bin/env python3
"""
全面性能测试脚本 - 验证events.py优化效果
"""

import time
import numpy as np
import torch
import sys
import os

# 添加路径
sys.path.append('/home/user/IsaacLab/source/isaaclab/isaaclab/envs/mdp')

def test_events_performance():
    """测试events.py中move_acceleration函数的性能"""
    print("=== Events.py 性能测试 ===")
    
    # 模拟环境参数
    num_envs = 1024
    num_steps = 1000
    
    print(f"测试环境数量: {num_envs}")
    print(f"测试步数: {num_steps}")
    
    # 模拟环境对象
    class MockEnv:
        def __init__(self):
            self._vehicle_dict = {}
            self._comparison_data = {}
            self._debug_counter = 0
            self._save_dir_created = False
            self._sim_step_counter = 0
    
    # 模拟资产对象
    class MockAsset:
        def __init__(self, num_envs):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # 模拟数据
            self.data = type('obj', (object,), {})()
            self.data.root_pos_w = torch.randn(num_envs, 3, device=self.device)
            self.data.root_quat_w = torch.randn(num_envs, 4, device=self.device)
            self.data.root_lin_vel_b = torch.randn(num_envs, 3, device=self.device)
            self.data.root_ang_vel_b = torch.randn(num_envs, 3, device=self.device)
            self.data.root_ang_vel_w = torch.randn(num_envs, 3, device=self.device)
            self.data.root_lin_vel_w = torch.randn(num_envs, 3, device=self.device)
            self.data.body_lin_acc_w = torch.randn(num_envs, 3, device=self.device)
            self.data.body_ang_acc_w = torch.randn(num_envs, 3, device=self.device)
            self.data.default_mass = torch.ones(num_envs, device=self.device) * 1000
            self.data.default_inertia = torch.ones(num_envs, 9, device=self.device)
    
    # 创建模拟对象
    env = MockEnv()
    asset = MockAsset(num_envs)
    env_ids = torch.arange(num_envs, device=asset.device)
    
    # 导入船舶控制系统
    from vessels import VesselControlSystem
    
    # 初始化船舶系统
    start_time = time.time()
    for i in range(num_envs):
        initial_eta = np.random.normal(0, 0.1, 6)
        initial_nu = np.random.normal(0, 0.1, 6)
        env._vehicle_dict[i] = VesselControlSystem(
            target_position=[10, 10, np.pi],
            initial_eta=initial_eta,
            initial_nu=initial_nu,
            dt=0.02
        )
    
    init_time = time.time() - start_time
    print(f"初始化{num_envs}个船舶系统耗时: {init_time:.2f}秒")
    
    # 模拟move_acceleration函数的核心逻辑
    def simulate_move_acceleration(env, asset, env_ids, step):
        """模拟move_acceleration函数的核心计算"""
        dt = 0.02
        time_me = step * 0.25
        
        # 模拟位置和速度数据
        current_quat = asset.data.root_quat_w[env_ids]
        current_pos = asset.data.root_pos_w[env_ids]
        nu_lin = asset.data.root_lin_vel_b[env_ids]
        nu_ang = asset.data.root_ang_vel_b[env_ids]
        nu = torch.cat([nu_lin, nu_ang], dim=-1)
        
        # 计算相对位置和角度
        rot_angles = torch.stack([
            torch.atan2(2 * (current_quat[:, 0] * current_quat[:, 1] + current_quat[:, 2] * current_quat[:, 3]),
                       1 - 2 * (current_quat[:, 1]**2 + current_quat[:, 2]**2)),
            torch.asin(2 * (current_quat[:, 0] * current_quat[:, 2] - current_quat[:, 3] * current_quat[:, 1])),
            torch.atan2(2 * (current_quat[:, 0] * current_quat[:, 3] + current_quat[:, 1] * current_quat[:, 2]),
                       1 - 2 * (current_quat[:, 2]**2 + current_quat[:, 3]**2))
        ], dim=1)
        
        pose = torch.cat([current_pos, rot_angles], dim=1)
        
        # 为每个环境计算加速度
        acc_list = []
        for i, env_id in enumerate(env_ids.tolist()):
            current_pose = pose[i]
            current_nu = nu[i]
            
            # 调用船舶控制系统
            acc, eta_dot = env._vehicle_dict[env_id].step(current_pose, current_nu, time_me * dt)
            acc_list.append(acc)
        
        # 堆叠加速度
        nu_dot = torch.stack(acc_list, dim=0)
        
        # 模拟力和力矩计算
        lin_acc = nu_dot[:, :3]
        ang_acc = nu_dot[:, 3:]
        
        # 获取质量和惯性
        mass = asset.data.default_mass.to(asset.device)[env_ids].unsqueeze(-1)
        inertia = asset.data.default_inertia.to(asset.device)[env_ids]
        inertia_mat = inertia.view(-1, 3, 3)
        
        # 计算力和力矩
        force = mass * lin_acc.unsqueeze(1)
        torque = torch.bmm(inertia_mat, ang_acc.unsqueeze(-1)).squeeze(-1)
        
        return force, torque
    
    # 运行性能测试
    print("开始性能测试...")
    simulation_start = time.time()
    
    for step in range(num_steps):
        force, torque = simulate_move_acceleration(env, asset, env_ids, step)
        
        # 每100步显示进度
        if (step + 1) % 100 == 0:
            elapsed = time.time() - simulation_start
            print(f"完成 {step + 1}/{num_steps} 步, 耗时: {elapsed:.2f}秒")
    
    total_time = time.time() - simulation_start
    
    print(f"\n=== 性能测试结果 ===")
    print(f"仿真总耗时: {total_time:.2f}秒")
    print(f"平均每步耗时: {total_time / num_steps * 1000:.2f}毫秒")
    print(f"每秒可处理步数: {num_steps / total_time:.0f}")
    print(f"每步每环境耗时: {total_time / (num_steps * num_envs) * 1000:.3f}毫秒")

def test_memory_efficiency():
    """测试内存效率"""
    import psutil
    import gc
    
    print("\n=== 内存效率测试 ===")
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"初始内存使用: {initial_memory:.1f} MB")
    
    # 创建大量船舶系统
    from vessels import VesselControlSystem
    
    vessel_systems = []
    for i in range(100):
        vessel_system = VesselControlSystem(
            target_position=[10, 10, np.pi],
            initial_eta=np.random.normal(0, 0.1, 6),
            initial_nu=np.random.normal(0, 0.1, 6),
            dt=0.02
        )
        vessel_systems.append(vessel_system)
    
    after_creation_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"创建100个系统后内存使用: {after_creation_memory:.1f} MB")
    print(f"每个系统平均内存: {(after_creation_memory - initial_memory) / 100:.1f} MB")
    
    # 清理
    del vessel_systems
    gc.collect()
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"清理后内存使用: {final_memory:.1f} MB")

if __name__ == "__main__":
    try:
        test_events_performance()
        test_memory_efficiency()
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
