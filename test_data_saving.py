#!/usr/bin/env python3
"""
测试数据保存功能
"""

import numpy as np
import os

def test_save_comparison_data():
    """测试保存对比数据功能"""
    
    # 模拟环境对象
    class MockEnv:
        def __init__(self):
            self._comparison_data = {}
    
    # 创建模拟环境
    env = MockEnv()
    env_id = 0
    
    # 初始化对比数据
    env._comparison_data[env_id] = {
        'isaaclab_eta_history': [],
        'isaaclab_nu_history': [],
        'calculated_eta_history': [],
        'calculated_nu_history': [],
        'calculated_eta': np.zeros(6),
        'calculated_nu': np.zeros(6),
        'step_count': 0
    }
    
    # 模拟一些数据
    comp_data = env._comparison_data[env_id]
    for i in range(5):
        comp_data['step_count'] += 1
        comp_data['isaaclab_eta_history'].append(np.random.randn(6))
        comp_data['isaaclab_nu_history'].append(np.random.randn(6))
        comp_data['calculated_eta_history'].append(np.random.randn(6))
        comp_data['calculated_nu_history'].append(np.random.randn(6))
    
    print(f"模拟数据准备完成 - 步数: {comp_data['step_count']}, 历史长度: {len(comp_data['isaaclab_eta_history'])}")
    
    # 测试保存函数
    def save_comparison_data(env, env_id):
        """保存对比数据到文件"""
        import numpy as np
        import os
        
        if not hasattr(env, '_comparison_data') or env_id not in env._comparison_data:
            print(f"[ERROR] 环境 {env_id} 没有对比数据")
            return
        
        comp_data = env._comparison_data[env_id]
        
        # 创建保存目录
        save_dir = "/home/user/IsaacLab/comparison_data"
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存数据
        filename = f"{save_dir}/env_{env_id}_step_{comp_data['step_count']}.npz"
        np.savez(filename,
                 isaaclab_eta_history=np.array(comp_data['isaaclab_eta_history']),
                 isaaclab_nu_history=np.array(comp_data['isaaclab_nu_history']),
                 calculated_eta_history=np.array(comp_data['calculated_eta_history']),
                 calculated_nu_history=np.array(comp_data['calculated_nu_history']),
                 step_count=comp_data['step_count'])
        
        print(f"对比数据已保存到: {filename}")
        
        # 验证文件是否存在
        if os.path.exists(filename):
            print(f"✓ 文件保存成功: {filename}")
            # 检查文件大小
            file_size = os.path.getsize(filename)
            print(f"✓ 文件大小: {file_size} 字节")
        else:
            print(f"✗ 文件保存失败: {filename}")
    
    # 执行保存
    save_comparison_data(env, env_id)
    
    # 检查保存目录
    save_dir = "/home/user/IsaacLab/comparison_data"
    if os.path.exists(save_dir):
        files = os.listdir(save_dir)
        print(f"✓ 保存目录存在: {save_dir}")
        print(f"✓ 目录中的文件: {files}")
    else:
        print(f"✗ 保存目录不存在: {save_dir}")

if __name__ == "__main__":
    test_save_comparison_data()
