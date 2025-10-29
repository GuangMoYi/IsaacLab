import numpy as np
import math
import torch
from typing import Tuple

class OptimizedVesselControlSystem:
    """
    优化版船舶控制系统 - 大幅简化以提高性能
    专门用于IsaacLab环境中的快速训练
    """
    
    def __init__(self, target_position=None, initial_eta=None, initial_nu=None, dt=0.02):
        """
        初始化优化版船舶控制系统
        
        参数:
            target_position: 期望位置 [x, y, yaw]，默认为 [10, 10, π]
            initial_eta: 初始位置 [x, y, z, roll, pitch, yaw]
            initial_nu: 初始速度 [u, v, w, p, q, r]
        """
        self.dt = dt
        
        # 简化的控制参数 - 大幅降低计算复杂度
        self.Kp = np.diag([1e4, 1e4, 1e6])  # 大幅降低比例增益
        self.Kd = np.diag([1e2, 1e2, 1e3])  # 大幅降低阻尼增益
        
        # 简化的质量矩阵 - 使用对角矩阵
        self.M = np.diag([1000, 1000, 1000, 100, 100, 100])
        self.inv_M = np.linalg.inv(self.M)
        
        # 简化的阻尼矩阵
        self.D = np.diag([100, 100, 100, 10, 10, 10])
        
        # 简化的重力矩阵
        self.G = np.zeros((6, 6))
        
        # 设置期望位置
        if target_position is None:
            self.target_position = np.array([10, 10, np.pi])
        else:
            self.target_position = np.array(target_position)
        
        # 初始化状态
        if initial_eta is None:
            self.eta = np.array([0, 0, 0, 0, 0, 0], dtype=float)
        else:
            self.eta = np.array(initial_eta, dtype=float)
            
        if initial_nu is None:
            self.nu = np.zeros(6, dtype=float)
        else:
            self.nu = np.array(initial_nu, dtype=float)
        
        # 简化的参考轨迹
        self.reference = np.array([self.eta[0], self.eta[1], self.eta[5], 0, 0, 0])
        
        # 简化的波浪载荷 - 使用简单的正弦波
        self.wave_amplitude = 0.1  # 大幅降低波浪幅度
        self.wave_frequency = 0.5  # 降低波浪频率
        
    def Rzyx(self, euler: np.ndarray) -> np.ndarray:
        """简化的旋转矩阵计算"""
        phi, theta, psi = euler
        cpsi, spsi = np.cos(psi), np.sin(psi)
        ctheta, stheta = np.cos(theta), np.sin(theta)  
        cphi, sphi = np.cos(phi), np.sin(phi)
        
        Rz = np.array([[cpsi, -spsi, 0], [spsi, cpsi, 0], [0, 0, 1]])
        Ry = np.array([[ctheta, 0, stheta], [0, 1, 0], [-stheta, 0, ctheta]])
        Rx = np.array([[1, 0, 0], [0, cphi, -sphi], [0, sphi, cphi]])
        
        return Rz @ Ry @ Rx
    
    def generate_simple_wave_loads(self, t):
        """简化的波浪载荷生成 - 使用简单的正弦波"""
        # 只生成主要的波浪载荷分量
        wave_loads = np.zeros(6)
        wave_loads[0] = self.wave_amplitude * np.sin(self.wave_frequency * t)  # surge
        wave_loads[1] = self.wave_amplitude * np.cos(self.wave_frequency * t)  # sway
        wave_loads[5] = 0.1 * self.wave_amplitude * np.sin(self.wave_frequency * t * 0.5)  # yaw
        
        return wave_loads
    
    def step(self, current_eta, current_nu, current_time: float):
        """
        优化版单步计算函数 - 大幅简化计算
        
        输入:
            current_eta: 当前时刻的位置 [x, y, z, roll, pitch, yaw]
            current_nu: 当前时刻的速度 [u, v, w, p, q, r]
            current_time: 当前时间
            
        输出:
            control_acceleration: 控制加速度nu_dot
            eta_dot: 位置导数
        """
        # 处理输入数据类型转换
        if hasattr(current_eta, 'cpu'):  # 如果是torch张量
            current_eta_np = current_eta.detach().cpu().numpy()
            current_nu_np = current_nu.detach().cpu().numpy()
            is_tensor = True
        else:  # 如果是numpy数组
            current_eta_np = current_eta
            current_nu_np = current_nu
            is_tensor = False
            
        # 更新内部状态
        self.eta = current_eta_np.copy()
        self.nu = current_nu_np.copy()
        
        # 简化的控制器 - 只控制x, y, yaw
        eta_3dof = self.eta[[0, 1, 5]]  # x, y, yaw
        nu_3dof = self.nu[[0, 1, 5]]    # u, v, r
        
        # 计算误差
        error = eta_3dof - self.target_position
        
        # 简化的控制律
        R = self.Rzyx(np.array([0, 0, eta_3dof[2]]))
        u_3dof = -R.T @ (self.Kp @ error + self.Kd @ R @ nu_3dof)
        
        # 将3DOF控制力转换为6DOF推力
        tau_thruster = np.array([u_3dof[0], u_3dof[1], 0, 0, 0, u_3dof[2]])
        
        # 简化的波浪载荷
        wave_loads = self.generate_simple_wave_loads(current_time)
        
        # 计算总加速度 - 大幅简化
        nu_dot = self.inv_M @ (tau_thruster - self.D @ self.nu + wave_loads)
        
        # 计算位置导数
        R_full = self.Rzyx(self.eta[3:6])
        eta_dot = np.concatenate([R_full @ self.nu[:3], self.nu[3:6]])
        
        # 根据输入类型返回相应格式的结果
        if is_tensor:
            import torch
            return torch.from_numpy(nu_dot).to(
                dtype=current_eta.dtype, device=current_eta.device
            ), torch.from_numpy(eta_dot).to(
                dtype=current_eta.dtype, device=current_eta.device
            )
        else:
            return nu_dot, eta_dot
