import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import time
from typing import Tuple, Dict, Any
import os
from scipy.interpolate import interp1d

class VesselControlSystem:
    def __init__(self, target_position=None, initial_eta=None, initial_nu=None):
        """
        初始化船舶控制系统
        
        参数:
            target_position: 期望位置 [x, y, yaw]，默认为 [10, 10, π]
            initial_eta: 初始位置 [x, y, z, roll, pitch, yaw]，默认为 [0, 0, 0, 10°, 0, 0]
            initial_nu: 初始速度 [u, v, w, p, q, r]，默认为 [0, 0, 0, 0, 0, 0]
        """
        self.dt = 0.02
        self.eta_r_ddot = np.zeros(3)
        self.omega_o = 0.8976 * np.array([0.1, 0.1, 0.1])
        self.omega_c = 1.2255 * self.omega_o
        self.DELTA = np.diag([1, 1, 1])
        
        # 预计算常用矩阵
        self.I3 = np.eye(3)
        self.OMEGA = np.diag(self.omega_o)
        self.OMEGA2 = self.OMEGA @ self.OMEGA
        self.OMEGA3 = self.OMEGA2 @ self.OMEGA
        
        self.load_vessel_data()
        
        # 设置期望位置
        if target_position is None:
            self.parameters_ref = np.array([10, 10, np.pi])
        else:
            self.parameters_ref = np.array(target_position)
        
        # 初始化状态，传入初始位置和速度
        self.initialize_states(initial_eta, initial_nu)
        
        self.initialize_memory_effect_systems()
        
        # 预计算RK4系数
        self.rk4_coeffs = np.array([1/6, 1/3, 1/3, 1/6])
        
    def load_vessel_data(self):
        try:
            vessel_data = io.loadmat('vessel.mat')
            vesselABC_data = io.loadmat('vesselABC.mat')

            self.vessel = vessel_data['vessel'][0, 0] if 'vessel' in vessel_data else (print("vessel数据未找到") or {})
            self.vesselABC = vesselABC_data['vesselABC'][0, 0] if 'vesselABC' in vesselABC_data else (print("vesselABC数据未找到") or {})
            self.inv_M = self.vesselABC['Minv'] if 'Minv' in self.vesselABC.dtype.names else (print("Minv数据未找到") or np.eye(6))
            self.G = self.vesselABC['G'] if 'G' in self.vesselABC.dtype.names else (print("G数据未找到") or np.zeros((6, 6)))
            self.D = self.vessel['Bv'][:, :, 0] if 'Bv' in self.vessel.dtype.names else (print("Bv数据未找到") or np.zeros((6, 6)))
            self.C = np.zeros((6, 6))
        except FileNotFoundError:
            print("警告: 未找到船舶数据文件，使用默认参数")
            self.vessel = type('obj', (object,), {})()
            self.vesselABC = type('obj', (object,), {})()
            self.inv_M = np.eye(6)
            self.G = np.zeros((6, 6))
            self.D = np.zeros((6, 6))
            self.C = np.zeros((6, 6))

    def initialize_states(self, initial_eta=None, initial_nu=None):
        """
        初始化系统状态
        
        参数:
            initial_eta: 初始位置 [x, y, z, roll, pitch, yaw]
            initial_nu: 初始速度 [u, v, w, p, q, r]
        """
        self.u = np.zeros(3)
        self.xi_hat = np.zeros(6)
        self.nu_hat = np.zeros(3)
        self.b_hat = np.zeros(3)
        
        # 设置初始位置
        if initial_eta is None:
            self.eta = np.array([0, 0, 0, 10 * np.pi / 180, 0, 0], dtype=float)
        else:
            self.eta = np.array(initial_eta, dtype=float)
        
        # 设置初始速度
        if initial_nu is None:
            self.nu = np.zeros(6, dtype=float)
        else:
            self.nu = np.array(initial_nu, dtype=float)
        
        # 初始化参考轨迹
        self.reference = np.array([self.eta[0], self.eta[1], self.eta[5], 0, 0, 0])
        self.x_hat = np.zeros(6)
        self.state = np.concatenate([self.eta, self.nu, self.reference, self.x_hat])
        
        # 预计算索引，避免重复切片
        self.idx_eta = slice(0, 6)
        self.idx_nu = slice(6, 12)
        self.idx_ref = slice(12, 18)
        self.idx_xhat = slice(18, 24)

    def initialize_memory_effect_systems(self):
        if not hasattr(self, 'vesselABC') or 'Ar' not in self.vesselABC.dtype.names:
            self.memory_systems = None
            print("没有找到vesselABC数据或Ar数据")
            return
            
        Ar = self.vesselABC['Ar']
        Br = self.vesselABC['Br']
        Cr = self.vesselABC['Cr']
        Dr = self.vesselABC['Dr']
        
        def safe_get_matrix(cell_array, i, j):
            try:
                matrix = cell_array[i, j]
                return matrix if matrix.size > 0 else np.array([])
            except (IndexError, AttributeError):
                return np.array([])

        # 使用列表存储系统参数，便于向量化处理
        self.memory_systems = []
        self.memory_states = []
        
        # 定义系统索引映射
        system_indices = [
            (0, 0), (0, 2), (0, 4),  # 系统1,2,3
            (1, 1), (1, 3), (1, 5),  # 系统4,5,6  
            (2, 0), (2, 2), (2, 4),  # 系统7,8,9
            (3, 1), (3, 3), (3, 5),  # 系统10,11,12
            (4, 0), (4, 2), (4, 4),  # 系统13,14,15
            (5, 1), (5, 3), (5, 5)   # 系统16,17,18
        ]
        
        for i, j in system_indices:
            A = safe_get_matrix(Ar, i, j)
            B = safe_get_matrix(Br, i, j)
            C = safe_get_matrix(Cr, i, j)
            D = safe_get_matrix(Dr, i, j)
            
            self.memory_systems.append((A, B, C, D))
            self.memory_states.append(np.zeros(A.shape[0]) if A.size > 0 else np.array([]))

    def calculate_memory_effects(self, nu_r: np.ndarray) -> np.ndarray:
        if self.memory_systems is None:
            print("没有memory_systems")
            return np.zeros(6)
            
        nu_components = [nu_r[0], nu_r[2], nu_r[4],  # u, w, q
                        nu_r[1], nu_r[3], nu_r[5],  # v, p, r
                        nu_r[0], nu_r[2], nu_r[4],  # u, w, q
                        nu_r[1], nu_r[3], nu_r[5],  # v, p, r
                        nu_r[0], nu_r[2], nu_r[4],  # u, w, q
                        nu_r[1], nu_r[3], nu_r[5]]  # v, p, r
        
        outputs = np.zeros(18)
        
        # 批量处理所有系统
        for idx, (u, (A, B, C, D)) in enumerate(zip(nu_components, self.memory_systems)):
            if A.size > 0:
                y, self.memory_states[idx] = self.Dp_system(self.memory_states[idx], u, A, B, C, D)
                outputs[idx] = y
        
        # 合并输出到6个自由度
        mef = np.array([
            outputs[0] + outputs[1] + outputs[2],    # 自由度1
            outputs[3] + outputs[4] + outputs[5],    # 自由度2  
            outputs[6] + outputs[7] + outputs[8],    # 自由度3
            outputs[9] + outputs[10] + outputs[11],  # 自由度4
            outputs[12] + outputs[13] + outputs[14], # 自由度5
            outputs[15] + outputs[16] + outputs[17]  # 自由度6
        ])
        
        return mef

    def Dp_system(self, x: np.ndarray, u: float, A: np.ndarray, B: np.ndarray,
                      C: np.ndarray, D: np.ndarray) -> Tuple[float, np.ndarray]:
        if A.size == 0:
            print("DP系统中A为空")
            return 0.0, np.array([])
            
        # 优化的RK4实现
        k1 = A @ x + B * u
        k2 = A @ (x + 0.5 * self.dt * k1) + B * u
        k3 = A @ (x + 0.5 * self.dt * k2) + B * u  
        k4 = A @ (x + self.dt * k3) + B * u
        
        x_next = x + self.dt * (self.rk4_coeffs[0] * k1 + self.rk4_coeffs[1] * k2 + 
                               self.rk4_coeffs[2] * k3 + self.rk4_coeffs[3] * k4)
        
        y = C @ x + D * u
        return y.flat[0], x_next

    def Rzyx(self, euler: np.ndarray) -> np.ndarray:
        phi, theta, psi = euler
        
        # 预计算三角函数
        cpsi, spsi = np.cos(psi), np.sin(psi)
        ctheta, stheta = np.cos(theta), np.sin(theta)  
        cphi, sphi = np.cos(phi), np.sin(phi)
        
        Rz = np.array([[cpsi, -spsi, 0], [spsi, cpsi, 0], [0, 0, 1]])
        Ry = np.array([[ctheta, 0, stheta], [0, 1, 0], [-stheta, 0, ctheta]])
        Rx = np.array([[1, 0, 0], [0, cphi, -sphi], [0, sphi, cphi]])
        
        return Rz @ Ry @ Rx

    def T_Theta(self, Theta: np.ndarray) -> np.ndarray:
        phi, theta, psi = Theta
        
        # 预计算三角函数
        ct = np.cos(theta)
        st = np.sin(theta)
        sp = np.sin(phi)
        cp = np.cos(phi)
        
        # 避免除零
        epsilon = 1e-10
        if abs(ct) < epsilon:
            ct = np.sign(ct) * epsilon
            
        T = np.array([
            [1, sp * st/ct, cp * st/ct],
            [0, cp, -sp],
            [0, sp/ct, cp/ct]
        ])
        return T

    def reference_model_dynamics(self, eta_r: np.ndarray, eta_r_dot: np.ndarray) -> np.ndarray:
        x0 = np.concatenate([eta_r, eta_r_dot, self.eta_r_ddot])
        x_next = self.rk4_fast(self.reference_model_state_space, x0, self.parameters_ref)
        self.eta_r_ddot = x_next[6:9]
        return self.eta_r_ddot

    def reference_model_state_space(self, x: np.ndarray, parameters_ref: np.ndarray) -> np.ndarray:
        x1, x2, x3 = x[0:3], x[3:6], x[6:9]
        
        # 使用预计算的矩阵
        dx1 = x2
        dx2 = x3
        dx3 = -(2 * self.DELTA + self.I3) @ self.OMEGA @ x3 - \
               (2 * self.DELTA + self.I3) @ self.OMEGA2 @ x2 - \
               self.OMEGA3 @ x1 + self.OMEGA3 @ parameters_ref
               
        return np.concatenate([dx1, dx2, dx3])

    def observer_dynamics(self, u: np.ndarray, y: np.ndarray, y_hat: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        Cw = np.block([np.diag([0, 0, 0]), np.diag([1, 1, 1])])
        eta_hat = y_hat - Cw @ self.xi_hat
        
        x0 = np.concatenate([self.xi_hat, eta_hat, self.b_hat, self.nu_hat])
        x_next = self.rk4_fast(self.observer_dynamics_rhs, x0, u, y, y_hat)
        
        self.xi_hat = x_next[0:6]
        eta_hat = x_next[6:9]
        self.b_hat = x_next[9:12]
        self.nu_hat = x_next[12:15]
        
        x_hat = np.concatenate([eta_hat, self.nu_hat])
        return x_hat, self.b_hat, self.xi_hat, self.nu_hat

    def observer_dynamics_rhs(self, x: np.ndarray, u: np.ndarray, y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
        xi_hat, eta_hat, b_hat, nu_hat = x[0:6], x[6:9], x[9:12], x[12:15]
        
        # 预计算增益矩阵
        if not hasattr(self, '_K_precomputed'):
            # print("初始化K_precomputed")
            zeta_ni, lambda_ni = 1.0, 0.1
            K_11 = -2 * (zeta_ni - lambda_ni) * self.omega_c[0] / self.omega_o[0]
            K_12 = -2 * (zeta_ni - lambda_ni) * self.omega_c[1] / self.omega_o[1]
            K_13 = -2 * (zeta_ni - lambda_ni) * self.omega_c[2] / self.omega_o[2]
            K_14 = 2 * self.omega_o[0] * (zeta_ni - lambda_ni)
            K_15 = 2 * self.omega_o[1] * (zeta_ni - lambda_ni)
            K_16 = 2 * self.omega_o[2] * (zeta_ni - lambda_ni)
            
            self.K1 = np.vstack([
                np.diag([K_11, K_12, K_13]),
                np.diag([K_14, K_15, K_16])
            ])
            self.K2 = np.diag(self.omega_c)
            self.K4 = 1e3 * np.diag([0.1, 0.1, 0.001])
            self.K3 = 0.05 * self.K4
            self.Aw = np.block([
                [np.zeros((3, 3)), np.eye(3)],
                [-self.OMEGA2, -2 * self.DELTA @ self.OMEGA]
            ])
            self.T = 1000 * np.eye(3)
            self.invT = np.linalg.inv(self.T)
            
            # 修正：确保使用正确的维度
            self.invM_reduced = self.inv_M[np.ix_([0,1,5], [0,1,5])] if hasattr(self, 'vesselABC') and hasattr(self, 'inv_M') else (print("Minv数据未找到") or np.eye(3))
            self.D_reduced = self.D[np.ix_([0,1,5], [0,1,5])] if hasattr(self, 'D') else (print("D数据未找到") or np.zeros((3, 3)))
            
            self._K_precomputed = True
        
        y_tilde = y - y_hat
        R = self.Rzyx(np.array([0, 0, y[2]]))
        
        # 修正：确保维度匹配
        xi_hat_dot = self.Aw @ xi_hat + self.K1 @ y_tilde
        eta_hat_dot = R @ nu_hat + self.K2 @ y_tilde
        b_hat_dot = -self.invT @ b_hat + self.K3 @ y_tilde
        
        # 修正：确保所有矩阵都是3x3维度
        nu_hat_dot = self.invM_reduced @ (
            -self.D_reduced @ nu_hat + 
            R.T @ b_hat + 
            u + 
            R.T @ self.K4 @ y_tilde
        )
        
        return np.concatenate([xi_hat_dot, eta_hat_dot, b_hat_dot, nu_hat_dot])

    def controller(self, eta_r: np.ndarray, x_hat: np.ndarray, b_hat: np.ndarray) -> np.ndarray:
        """优化的控制器 - 返回控制力"""
        if not hasattr(self, '_controller_gains'):
            self.Kp = 1e5 * np.diag([2e3, 2e3, 1e6])
            self.Kd = 0 * np.diag([1e1, 1e1, 1e1])
            self._controller_gains = True
            
        eta_hat, nu_hat = x_hat[0:3], x_hat[3:6]
        error = eta_hat - eta_r
        R = self.Rzyx(np.array([0, 0, eta_hat[2]]))
        
        u = -R.T @ (self.Kp @ error + self.Kd @ R @ nu_hat + b_hat)
        return u
    
    def controller_acceleration(self, eta_r: np.ndarray, x_hat: np.ndarray, b_hat: np.ndarray, 
                               current_eta: np.ndarray, current_nu: np.ndarray) -> np.ndarray:
        """控制器 - 返回控制加速度nu_dot"""
        if not hasattr(self, '_controller_gains'):
            self.Kp = 1e5 * np.diag([2e3, 2e3, 1e6])
            self.Kd = 0 * np.diag([1e1, 1e1, 1e1])
            self._controller_gains = True
            
        eta_hat, nu_hat = x_hat[0:3], x_hat[3:6]
        error = eta_hat - eta_r
        R = self.Rzyx(np.array([0, 0, eta_hat[2]]))
        
        # 计算控制力
        u = -R.T @ (self.Kp @ error + self.Kd @ R @ nu_hat + b_hat)
        
        # 将控制力转换为6DOF推力
        tau_thruster = np.array([u[0], u[1], 0, 0, 0, u[2]])
        
        # 计算其他力
        nu_r = current_nu
        tau_cf = self.crossflow_drag(nu_r)
        mef = self.calculate_memory_effects(nu_r)
        damping_force = self.D @ nu_r
        gravity_force = self.G @ current_eta
        
        # 计算总加速度
        nu_dot = self.inv_M @ (tau_thruster - self.C @ nu_r - damping_force - 
                             gravity_force + tau_cf - mef)
        
        return nu_dot

    def crossflow_drag(self, nu_r: np.ndarray) -> np.ndarray:
        """优化的横流阻力计算"""
        if not hasattr(self, '_drag_params_init'):
            if 'main' in self.vessel.dtype.names:
                main_data = self.vessel['main'][0,0]
                T = main_data['T'][0,0]
                B = main_data['B'][0,0]
                Lpp_get = main_data['Lpp'][0,0]
            else:
                T = self.vessel['T'][0,0] if 'T' in self.vessel.dtype.names else 10
                B = self.vessel['B'][0,0] if 'B' in self.vessel.dtype.names else 30
                Lpp_get = self.vessel['Lpp'][0,0] if 'Lpp' in self.vessel.dtype.names else 200
                
            self._Cx = 1
            self._Ax = 0.9 * T * B  
            self._Ay = 0.9 * T * Lpp_get  
            self._CD = self.Hoerner(B, T)
            
            N = 20
            Lpp = 200
            dx = Lpp / (N - 1)
            Lpp2 = Lpp / 2
            self._x_points = np.arange(N) * dx - Lpp2
            self._weights = np.ones(N) * dx
            self._weights[0] = self._weights[-1] = 0.5 * dx
            
            # 预计算常数
            self._rho_half = 0.5 * 1025
            self._Ay_scale = self._Ay / 200
            self._drag_params_init = True
            
        u_r, v_r, r = nu_r[0], nu_r[1], nu_r[5]
        
        # 向量化计算
        v_local = np.clip(v_r + self._x_points * r, -100, 100)
        f_values = v_local * np.abs(v_local)
        weighted_f = f_values * self._weights
        
        sum1 = np.sum(weighted_f)
        sum2 = np.sum(weighted_f * self._x_points)
        
        X_drag = -self._Ax * self._Cx * self._rho_half * abs(u_r) * u_r
        Y_drag = -self._Ay_scale * self._CD * self._rho_half * sum1
        Z_drag = -self._Ay_scale * self._CD * self._rho_half * sum2
        
        return np.array([X_drag, Y_drag, 0, 0, 0, Z_drag])

    def Hoerner(self, B: float, T: float) -> float:
        CD_DATA = np.array([
            [0.0108623, 1.96608], [0.176606, 1.96573], [0.353025, 1.89756],
            [0.451863, 1.78718], [0.472838, 1.58374], [0.492877, 1.27862],
            [0.493252, 1.21082], [0.558473, 1.08356], [0.646401, 0.998631],
            [0.833589, 0.87959], [0.988002, 0.828415], [1.30807, 0.759941],
            [1.63918, 0.691442], [1.85998, 0.657076], [2.31288, 0.630693],
            [2.59998, 0.596186], [3.00877, 0.586846], [3.45075, 0.585909],
            [3.7379, 0.559877], [4.00309, 0.559315]
        ])
        ratio = B / (2 * T)
        return np.interp(ratio, CD_DATA[:, 0], CD_DATA[:, 1])

    def rk4_fast(self, func, x0, *args) -> np.ndarray:
        """优化的RK4实现"""
        k1 = func(x0, *args)
        k2 = func(x0 + 0.5 * self.dt * k1, *args)
        k3 = func(x0 + 0.5 * self.dt * k2, *args)
        k4 = func(x0 + self.dt * k3, *args)
        
        return x0 + self.dt * (self.rk4_coeffs[0] * k1 + self.rk4_coeffs[1] * k2 + 
                              self.rk4_coeffs[2] * k3 + self.rk4_coeffs[3] * k4)

    def step(self, current_eta: np.ndarray, current_nu: np.ndarray, current_time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        单步计算函数
        
        输入:
            current_eta: 当前时刻的位置 [x, y, z, roll, pitch, yaw]
            current_nu: 当前时刻的速度 [u, v, w, p, q, r]
            current_time: 当前时间
            
        输出:
            next_eta: 下一个时刻的位置
            next_nu: 下一个时刻的速度  
            current_control_acceleration: 当前时刻的控制加速度nu_dot
        """
        # 更新内部状态
        self.eta = current_eta.copy()
        self.nu = current_nu.copy()
        
        # 生成测量噪声
        np.random.seed(int(current_time * 1000) % 2**32)  # 基于时间的随机种子
        noise_eta = np.random.normal(0, 0.001, 3)
        y = self.eta[[0, 1, 5]] + noise_eta
        
        # 参考轨迹计算
        eta_r, eta_r_dot = self.reference[0:3], self.reference[3:6]
        self.eta_r_ddot = self.reference_model_dynamics(eta_r, eta_r_dot)
        reference_dot = np.concatenate([eta_r_dot, self.eta_r_ddot])
        
        # 观测器更新
        y_hat = self.x_hat[0:3]
        self.x_hat, self.b_hat, self.xi_hat, self.nu_hat = self.observer_dynamics(self.u, y, y_hat)
        self.x_hat = np.concatenate([self.eta[[0, 1, 5]], self.nu[[0, 1, 5]]])
        
        # 控制器计算 - 返回控制加速度
        current_control_acceleration = self.controller_acceleration(eta_r, self.x_hat, self.b_hat, 
                                                                   current_eta, current_nu)
        
        # 添加波浪载荷
        wave_loads = self.generate_wave_loads_jonswap(current_time)
        current_control_acceleration += self.inv_M @ wave_loads
        
        # 计算位置导数
        R = self.Rzyx(self.eta[3:6])
        T_mat = self.T_Theta(self.eta[3:6])
        eta_dot = np.concatenate([R @ self.nu[:3], T_mat @ self.nu[3:6]])
        
        # 状态更新
        next_eta = self.eta + eta_dot * self.dt
        next_nu = self.nu + current_control_acceleration * self.dt
        
        # 更新参考轨迹
        self.reference += reference_dot * self.dt
        self.state = np.concatenate([next_eta, next_nu, self.reference, self.x_hat])
        
        return next_eta, next_nu, current_control_acceleration
    
    def generate_wave_loads_jonswap(self, t):
        """
        极致优化版波浪载荷生成函数
        保持结果完全一致，仅加速计算
        """
        if not hasattr(self, '_wave_init'):
            Hs = 1
            Tp = 8
            g = 9.81
            omega_p = 2 * np.pi / Tp
            gamma = 3.3
            
            vessel = self.vessel
            forceRAO = vessel['forceRAO'][0, 0]
            w = forceRAO['w'].flatten()
            w_min, w_max = np.min(w), np.max(w)
            Nw = 50
            self._wave_omega = np.linspace(w_min, w_max, Nw)
            domega = self._wave_omega[1] - self._wave_omega[0]
            
            # ---- JONSWAP 谱计算 ----
            sigma = np.where(self._wave_omega <= omega_p, 0.07, 0.09)
            S0 = (g**2 / self._wave_omega**5) * np.exp(-1.25 * (omega_p / self._wave_omega)**4) * \
                gamma**np.exp(-((self._wave_omega - omega_p)**2) / (2 * (sigma * omega_p)**2))
            alpha = Hs**2 / (16 * np.sum(S0 * domega))
            S = alpha * S0
            self._wave_spectrum_weight = np.sqrt(2 * S * domega)
            
            # ---- 固定随机相位 ----
            # np.random.seed(42)
            self._wave_epsilon = 0 * 2 * np.pi * np.random.rand(6, Nw)
            
            # ---- 预加载所有 DOF 数据 ----
            all_amp, all_phase, dof_sizes = [], [], []
            for d in range(6):
                amp = forceRAO['amp'][0, d]
                phase = forceRAO['phase'][0, d]
                ND, NM = amp.shape[1], amp.shape[2]
                all_amp.append(amp.reshape(-1, ND * NM))
                all_phase.append(phase.reshape(-1, ND * NM))
                dof_sizes.append(ND * NM)
            
            self._wave_amp_all = np.concatenate(all_amp, axis=1)
            self._wave_phase_all = np.concatenate(all_phase, axis=1)
            
            f_amp = interp1d(w, self._wave_amp_all, kind='linear', axis=0, fill_value='extrapolate')
            f_phase = interp1d(w, self._wave_phase_all, kind='linear', axis=0, fill_value='extrapolate')
            self._wave_amp_interp = f_amp(self._wave_omega)
            self._wave_phase_interp = f_phase(self._wave_omega)
            
            self._wave_dof_boundaries = np.cumsum([0] + dof_sizes)
            self._wave_init = True
        
        # ---- 高频优化计算 ----
        omega_t = self._wave_omega * t  # shape (Nw,)
        weight = self._wave_spectrum_weight  # shape (Nw,)
        tau_wave = np.zeros(6)

        for d in range(6):
            s, e = self._wave_dof_boundaries[d], self._wave_dof_boundaries[d + 1]
            amp_d = self._wave_amp_interp[:, s:e]           # (Nw, M)
            phase_d = self._wave_phase_interp[:, s:e]       # (Nw, M)
            base_phase = omega_t[:, None] + self._wave_epsilon[d, :, None]  # (Nw, 1)
            total_phase = base_phase + phase_d              # (Nw, M)
            cos_val = np.cos(total_phase)
            
            # 合并两次求和为 einsum（最优）
            tau_wave[d] = np.einsum('i,ij,ij->', weight, amp_d, cos_val)

        return tau_wave


    def plot_trajectory_comparison(self, time_array, ETA, REF, save_path="trajectory_comparison.png"):
        """
        绘制各个维度的实际轨迹与预期轨迹对比图
        """
        # 创建结果目录
        os.makedirs('results', exist_ok=True)
        
        # 维度名称 - 修正：参考轨迹是x, y, yaw
        dim_names = ['Surge (x)', 'Sway (y)', 'Heave (z)', 'Roll (φ)', 'Pitch (θ)', 'Yaw (ψ)']
        dim_units = ['(m)', '(m)', '(m)', '(rad)', '(rad)', '(rad)']
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # 参考轨迹对应的维度索引：x(0), y(1), yaw(5)
        ref_dim_indices = [0, 1, 5]
        ref_dim_names = ['x', 'y', 'ψ']
        
        for i in range(6):
            ax = axes[i]
            
            # 绘制实际轨迹
            if i == 5:  # 对于航向角，转换为度数
                actual_traj = np.rad2deg(ETA[:, i])
                unit = '(deg)'
            else:
                actual_traj = ETA[:, i]
                unit = dim_units[i]
            
            ax.plot(time_array, actual_traj, 'b-', linewidth=2, label='Actual')
            
            # 绘制参考轨迹（只在x, y, yaw维度）
            if i in ref_dim_indices:
                ref_idx = ref_dim_indices.index(i)  # 找到在参考维度列表中的索引
                if i == 5:  # yaw角度转换为度数
                    ref_traj = np.rad2deg(REF[:, ref_idx])
                else:
                    ref_traj = REF[:, ref_idx]
                
                ax.plot(time_array, ref_traj, 'r--', linewidth=2, label='Reference')
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(f'{dim_names[i]} {unit}')
            ax.set_title(f'{dim_names[i]} Trajectory')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 添加跟踪误差（对于有参考轨迹的维度）
            if i in ref_dim_indices:
                ref_idx = ref_dim_indices.index(i)
                
                if i == 5:  # yaw角度
                    error = np.rad2deg(ETA[:, i] - REF[:, ref_idx])
                else:
                    error = ETA[:, i] - REF[:, ref_idx]
                
                # 在图上显示最大误差和均方根误差
                max_error = np.max(np.abs(error))
                rmse = np.sqrt(np.mean(error**2))
                ax.text(0.02, 0.98, f'Max Error: {max_error:.3f}{unit}\nRMSE: {rmse:.3f}{unit}', 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()

        # 计算总误差（只计算有参考轨迹的维度）
        total_error = 0
        for i, dim_idx in enumerate(ref_dim_indices):
            if dim_idx == 5:  # yaw角度
                error = ETA[:, dim_idx] - REF[:, i] # np.rad2deg(ETA[:, dim_idx]) - np.rad2deg(REF[:, i])
            else:
                error = ETA[:, dim_idx] - REF[:, i]
            total_error += np.sum(np.abs(error))
        
        print(f"总误差: {total_error:.3f}")
        
        # 保存图片
        if save_path:
            full_save_path = f'results/{save_path}'
            plt.savefig(full_save_path, dpi=300, bbox_inches='tight')
            print(f"轨迹对比图已保存至: {full_save_path}")
        
        # plt.show()
        
def main():
    print("=== 船舶控制系统单步仿真循环 ===")
    
    # 设置初始输入 - 位置和速度
    initial_eta = np.array([0, 0, 0, 0, 0, 0], dtype=float)  # 初始位置
    initial_nu = np.array([0, 0, 0, 0, 0, 0], dtype=float)      # 初始速度
    target_position = [10, 10, np.pi]  # 期望位置 [x, y, yaw]
    
    # 创建系统，使用初始输入来初始化系统内部状态
    vessel_system = VesselControlSystem(
        target_position=target_position,
        initial_eta=initial_eta,
        initial_nu=initial_nu
    )
    
    print(f"期望位置: {target_position}")
    print(f"初始输入位置: {initial_eta}")
    print(f"初始输入速度: {initial_nu}")
    print(f"系统内部位置: {vessel_system.eta}")
    print(f"系统内部速度: {vessel_system.nu}")
    
    # 仿真参数
    total_time = 500
    dt = vessel_system.dt
    num_steps = int(total_time / dt)
    time_array = np.arange(0, total_time, dt)
    
    # 预分配内存
    ETA = np.zeros((num_steps, 6))
    UU = np.zeros((num_steps, 3))
    REF = np.zeros((num_steps, 3))
    
    print(f"开始单步仿真循环...")
    print(f"总步数: {num_steps}, 时间步长: {dt}")
    
    # 使用系统内部状态作为初始输入
    current_eta = vessel_system.eta.copy()
    current_nu = vessel_system.nu.copy()
    current_time = 0.0
    
    # 预生成噪声 - 与原始版本保持一致
    np.random.seed(123)
    noise_eta = np.random.normal(0, 0.001, (num_steps, 3))
    
    for i, t in enumerate(time_array):
        if i >= num_steps:
            break
        
        # 单步计算 - 输入当前位置和速度
        next_eta, next_nu, control_acceleration = vessel_system.step(
            current_eta, current_nu, current_time
        )
        
        # 存储结果
        ETA[i] = current_eta
        UU[i] = control_acceleration[:3]  # 只存储前3个分量
        REF[i] = vessel_system.reference[0:3]
        
        # 更新输入 - 下一步的输入是当前步的输出
        current_eta = next_eta
        current_nu = next_nu
        current_time += dt
        
        # 每10000步显示一次进度
        if (i + 1) % 10000 == 0:
            print(f"完成步骤 {i + 1}/{num_steps}, 时间: {current_time:.2f}s")
            print(f"  当前输入位置: {current_eta[[0, 1, 5]]}")  # 只显示x, y, yaw
            print(f"  当前输入速度: {current_nu[[0, 1, 5]]}")   # 只显示u, v, r
    
    print("单步仿真循环完成！")
    
    # 绘制轨迹对比图
    vessel_system.plot_trajectory_comparison(time_array, ETA, REF, "trajectory_comparison_single_step.png")
    
    print("\\n=== 单步仿真循环使用说明 ===")
    print("1. 创建系统时指定初始状态: vessel = VesselControlSystem(target_position=[x,y,yaw], initial_eta=[...], initial_nu=[...])")
    print("2. 系统内部状态与初始输入保持一致")
    print("3. 循环调用: next_eta, next_nu, control_acceleration = vessel.step(current_eta, current_nu, current_time)")
    print("4. 更新输入: current_eta = next_eta, current_nu = next_nu")
    print("5. 每次输入都是位置和速度，输出也是位置和速度")

if __name__ == "__main__":
    main()