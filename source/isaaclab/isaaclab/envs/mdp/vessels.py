import numpy as np
import math
import sys
# =================================================frigate================================================================
class frigate:
    """
    frigate()                               Step input, rudder angle
    frigate('headingAutopilot',U,psi_d)     Heading autopilot
    
    Inputs:
        U: cruise speed (m/s)
        psi_d: desired heading angle (deg)                                           and desired heading(deg)
    """        
    def __init__(self, controlSystem = 'stepInput', U = 5.0, r = 0.0):
                                  
        if (controlSystem == 'headingAutopilot'):
            self.controlDescription = (
                'Heading autopilot, psi_d = ' 
                + str(r) 
                + ' deg'
                )
             
        else:  
            self.controlDescription = "Step input for delta" 
            controlSystem = 'stepInput'  
      
        # Check if the speed U is within the valid range
        if (U < 5.0 or U > 12.0):
            sys.exit('The speed U should be between 5-12 m/s')          
            
        self.ref = r
        self.controlMode = controlSystem
                    
        # Initialize the ship model
        self.name = "Frigate (see 'frigate.py' for more details)"
        self.L = 100.0        # Length      
        self.deltaMax = 3000   # max rudder angle (deg)  
        self.DdeltaMax = 1000  # max rudder rate (deg/s)        
        self.nu  = np.array([ U, 0, 0, 0, 0, 0],float)  # velocity vector  
        self.u_actual = np.array([0],float)             # control input vector              

        self.controls = ['Rudder angle (deg)']
        self.dimU = len(self.controls)
        
        # ROV Zefakkel (Van Amerongen 1982) 
        self.n1 = 1
        self.n3 = 0.4
        
        # interpolate to find K, T and n3 as a function of U
        U_data = np.array([ 6.0, 9.0, 12.0 ],float)
        K_data = np.array([ 0.08, 0.18, 0.23 ],float)
        T_data = np.array([ 20.0, 27.0, 21.0 ],float) 
        n3_data = np.array([ 0.4, 0.6, 0.3 ],float)         
        

        U = self.nu[0]
        self.K  = np.interp( U, U_data, K_data )
        self.T  = np.interp( U, U_data, T_data )
        self.n3 = np.interp( U, U_data, n3_data )
    
        # Heading autopilot  
        self.e_int = 0.0         # integral state, initial value
        self.wn = 0.4            # PID pole placement parameters
        self.zeta = 1
        
        # Reference model
        self.r_max = 1 * math.pi / 180   # maximum yaw rate 
        self.psi_d = 0                   # position, velocity and acc. states
        self.r_d = 0
        self.a_d = 0
        self.wn_d = self.wn / 5
        self.zeta_d = 1       

        
    def dynamics(self,eta,nu,u_actual,u_control,sampleTime):
        """
        [nu, u_actual] = dynamics(eta,nu,u_actual,u_control,sampleTime)
        integrates the ship equations of motion using Euler's method.
        """       

        # States and inputs
        delta_c = u_control[0]
        delta   = u_actual[0]   
        r       = nu[5]
        
        # Rudder angle saturation and dynamics
        if ( abs(delta) >= self.deltaMax * math.pi/180 ):
            delta = np.sign(delta) * self.deltaMax * math.pi/180

        delta_dot = delta_c - delta
        if ( abs(delta_dot) >= self.DdeltaMax * math.pi/180 ):
            delta_dot = np.sign(delta_dot) * self.DdeltaMax * math.pi/180
                    
        # Dynamics
        r_dot = (1 / self.T) * ( self.K * delta - self.n3 * r**3 - self.n1 * r )
        nu_dot = np.array( [0, 0, 0, 0, 0, r_dot], float)
        
        # Forward Euler integration [k+1]
        nu  = nu + sampleTime * nu_dot
        delta = delta + sampleTime * delta_dot
        
        u_actual = np.array([delta],float)          
        
        return nu_dot, u_actual
    
    
    def stepInput(self,t):
        """
        delta_c = stepInput(t) generates stern plane step inputs.
        """    
        delta_c = 20 * (math.pi/180)    
        if t > 30:
            delta_c = 10 * (math.pi/180) 
        if t > 50:
            delta_c = 0
            
        u_control = np.array([delta_c],float)   
         
        return u_control             


    def headingAutopilot(self,eta,nu,sampleTime):
        """
        u = headingAutopilot(eta,nu,sampleTime) is a PID controller 
        for automatic heading control based on pole placement.
        
        delta = (T/K) * a_d + (1/K) * rd 
               - Kp * ( ssa( psi-psi_d ) + Td * (r - r_d) + (1/Ti) * z )
        
        """                  
        psi = eta[5]                # yaw angle
        r = nu[5]                   # yaw rate
        e_psi = self.wrap_angle(psi - self.psi_d)    # yaw angle tracking error
        e_r   = r - self.r_d        # yaw rate tracking error
        psi_ref = self.ref * math.pi / 180  # yaw angle setpoint
    
        wn = self.wn                # PID natural frequency
        zeta = self.zeta            # PID natural relative damping factor
        wn_d = self.wn_d            # reference model natural frequency
        zeta_d = self.zeta_d        # reference model relative damping factor

        m = self.T / self.K            
        d = self.n1 / self.K   
        k = 0

        # PID feedback controller with 3rd-order reference model（生成轨迹，下一步的期望位置、速度、加速度）
        [delta, self.e_int, self.psi_d, self.r_d, self.a_d] = \
            PIDpolePlacement( self.e_int, e_psi, e_r, self.psi_d, self.r_d, self.a_d, \
            m, d, k, wn_d, zeta_d, wn, zeta, psi_ref, self.r_max, sampleTime )

        u_control = np.array([delta],float)   
         
        return u_control
    def wrap_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
def PIDpolePlacement(
    e_int,
    e_x,
    e_v,
    x_d,
    v_d,
    a_d,
    m,
    d,
    k,
    wn_d,
    zeta_d,
    wn,
    zeta,
    r,
    v_max,
    sampleTime,
    ):

    # PID gains based on pole placement （原始定义）
    Kp = m * wn ** 2.0 - k
    Kd = m * 2.0 * zeta * wn - d
    Ki = (wn / 10.0) * Kp

    # frigate 原始DP参数
    # Kp =  20.338983050847464
    # Kd =  96.61016949152544
    # Ki =  0.8135593220338986
    

    # PID control law
    u = -Kp * e_x - Kd * e_v - Ki * e_int

    # Integral error, Euler's method
    e_int += sampleTime * e_x

    # 3rd-order reference model for smooth position, velocity and acceleration
    [x_d, v_d, a_d] = refModel3(x_d, v_d, a_d, r, wn_d, zeta_d, v_max, sampleTime)

    return u, e_int, x_d, v_d, a_d

def refModel3(x_d, v_d, a_d, r, wn_d, zeta_d, v_max, sampleTime):

    # desired "jerk"
    j_d = wn_d**3 * (r -x_d) - (2*zeta_d+1) * wn_d**2 * v_d - (2*zeta_d+1) * wn_d * a_d

# Forward Euler integration
    x_d += sampleTime * v_d             # desired position
    v_d += sampleTime * a_d             # desired velocity
    a_d += sampleTime * j_d             # desired acceleration 
    
    # Velocity saturation
    if (v_d > v_max):
        v_d = v_max
    elif (v_d < -v_max): 
        v_d = -v_max    
    
    return x_d, v_d, a_d


# =================================================semisub================================================================
class semisub:
    """
    semisub()                                      Propeller step inputs 
    semisub('DPcontrol',x_d,y_d,psi_d,V_c,beta_c)  DP control system
    
    Inputs:
        x_d: desired x position (m)
        y_d: desired y position (m)
        psi_d: desired yaw angle (deg)
        V_c: current speed (m/s)
        beta_c: current direction (deg)
    """

    def __init__(
        self,
        controlSystem="stepInput",
        r_x=0.0,
        r_y=0.0,
        r_n=0.0,
        V_current=0.0,
        beta_current=0.0,
    ):
        
        # Constants
        D2R = math.pi / 180                 # deg2rad

        if controlSystem == "DPcontrol":
            self.controlDescription = (
                "Nonlinear DP control (x_d, y_d, psi_d) = ("
                + str(r_x)
                + " m, "
                + str(r_y)
                + " m, "
                + str(r_n)
                + " deg)"
            )

        else:
            self.controlDescription = "Step inputs n = [n1, n2, n3, n4, n5, n6]"
            controlSystem = "stepInput"

        self.ref = np.array([r_x, r_y, r_n * D2R], float)
        self.V_c = V_current
        self.beta_c = beta_current * D2R
        self.controlMode = controlSystem

        # Initialize the semisub model
        self.L = 84.6   # Length (m)
        self.T_n = 1.0  # propeller revolution time constants (s)
        self.n_max = np.array(                      # RPM saturation limits (N)  
            [160, 160, 160, 160, 250, 250], float
        )                                     
        self.nu = np.array([0, 0, 0, 0, 0, 0], float)        # velocity vector
        self.u_actual = np.array([0, 0, 0, 0, 0, 0], float)  # RPM inputs
        self.name = "Semisubmersible (see 'semisub.py' for more details)"

        self.controls = [
            "#1 Bow tunnel thruster (RPM)",
            "#2 Bow tunnel thruster (RPM)",
            "#3 Aft tunnel thruster (RPM)",
            "#4 Aft tunnel thruster (RPM)",
            "#5 Right poontoon main propeller (RPM)",
            "$6 Left pontoon main propeller (RPM)",
        ]
        self.dimU = len(self.controls)
        
        # Semisub model
        MRB = 1.0e10 * np.array(
            [
                [0.0027, 0, 0, 0, -0.0530, 0],
                [0, 0.0027, 0, 0.0530, 0, -0.0014],
                [0, 0, 0.0027, 0, 0.0014, 0],
                [0, 0.0530, 0, 3.4775, 0, -0.0265],
                [-0.0530, 0, 0.0014, 0, 3.8150, 0],
                [0, -0.0014, 0, -0.0265, 0, 3.7192],
            ],
            float,
        )

        MA = 1.0e10 * np.array(
            [
                [0.0017, 0, 0, 0, -0.0255, 0],
                [0, 0.0042, 0, 0.0365, 0, 0],
                [0, 0, 0.0021, 0, 0, 0],
                [0, 0.0365, 0, 1.3416, 0, 0],
                [-0.0255, 0, 0, 0, 2.2267, 0],
                [0, 0, 0, 0, 0, 3.2049],
            ],
            float,
        )

        self.D = 1.0e09 * np.array(
            [
                [0.0004, 0, 0, 0, -0.0085, 0],
                [0, 0.0003, 0, 0.0067, 0, -0.0002],
                [0, 0, 0.0034, 0, 0.0017, 0],
                [0, 0.0067, 0, 4.8841, 0, -0.0034],
                [-0.0085, 0, 0.0017, 0, 7.1383, 0],
                [0, -0.0002, 0, -0.0034, 0, 0.8656],
            ],
            float,
        )

        self.G = 1.0e10 * np.diag([0.0, 0.0, 0.0006, 1.4296, 2.6212, 0.0])

        self.M = MRB + MA

        self.Minv = np.linalg.inv(self.M)

        # Thrust coefficient and configuration matrices (Fossen 2021, Ch. 11.2)
        K = np.diag([3.5, 3.5, 25.0, 25.0, 25.0, 25.0])
        T = np.array(
            [
                [0, 0, 0, 0, 1, 1],
                [1, 1, 1, 1, 0, 0],
                [30, 20, -20, -30, -self.L / 2, self.L / 2],
            ],
            float,
        )
        self.B = T @ K

        # DP control system
        self.e_int = np.array([0, 0, 0], float)  # integral states
        self.x_d = 0.0  # setpoints
        self.y_d = 0.0
        self.psi_d = 0.0
        self.wn = np.diag([0.15, 0.15, 0.05])  # PID pole placement
        self.zeta = np.diag([1.0, 1.0, 1.0])

        # GMY
        self.wave = 0
        self.eta = np.array([0, 0, 0, 0, 0, 0], float)
        # self.nu = np.array([0, 0, 0, 0, 0, 0], float)

        self.N_wave_freq = 100  # 频率分量个数
        self.omega_list = np.linspace(0.2, 2.5, self.N_wave_freq)  # rad/s   max: 2.5
        self.phase_list = 2 * np.pi * np.random.rand(self.N_wave_freq)  # 随机相位
        self.H_wave = 10.0  # 浪高
        self.T_wave = 5.0  # 浪周期


    def dynamics(self, eta, nu, u_actual, u_control, sampleTime, time_me):
        """
        [nu,u_actual] = dynamics(eta,nu,u_actual,u_control,sampleTime) integrates the
        semisub equations of motion using Euler's method.
        """

        # Input vector
        n = u_actual

        # Current velocities
        x_c = self.V_c * math.sin(eta[0] )  # current x velocity
        y_c = self.V_c * math.sin(eta[1] )  # current y velocity
        z_c = self.V_c * math.sin(eta[2] - self.beta_c)  # current z velocity
        u_c = self.V_c * math.sin(self.beta_c - eta[3])  # current surge velocity   eta[5]
        v_c = self.V_c * math.sin(self.beta_c - eta[4])  # current sway velocity    eta[5]
        w_c = self.V_c * math.sin(self.beta_c - eta[5])  # current yaw velocity  (更新还是慢了点)

        # 0 0 0的
        # x_c = 0.1 * self.V_c * math.sin(eta[0] )  # current x velocity
        # y_c = 0.1 * self.V_c * math.sin(eta[1] )  # current y velocity
        # z_c = 0.1 * self.V_c * math.sin(eta[2] )  # current z velocity
        # u_c = self.V_c * math.sin(self.beta_c - eta[3])  # current surge velocity   eta[5]
        # v_c = self.V_c * math.sin(self.beta_c - eta[4])  # current sway velocity    eta[5]
        # w_c = self.V_c * math.sin(self.beta_c - eta[5])  # current yaw velocity  (更新还是慢了点)
        
        nu_c = np.array([x_c, y_c, z_c, u_c, v_c, w_c], float)  # current velocity vector​
        nu_r = nu - nu_c  # relative velocity vector


        # Control forces and moments with propeller saturation
        n_squared = np.zeros(self.dimU)
        for i in range(0, self.dimU):
            n[i] = sat(n[i], -self.n_max[i], self.n_max[i])  # saturation limits
            n_squared[i] = abs(n[i]) * n[i]

        tau3 = np.matmul(self.B, n_squared)
        tau = np.array([tau3[0], tau3[1], 0, 0, 0, tau3[2]], float)

        # # # GMY 
        tau_wave = self.computeWaveForce_freq(time_me)
        tau = tau + tau_wave

        # 6-DOF semisub model
        nu_dot = np.matmul(
            self.Minv, tau - np.matmul(self.D, nu_r) - np.matmul(self.G, eta)
        )
        n_dot = (u_control - u_actual) / self.T_n
        
        # GMY
        # nu_dot = np.array([0, 0, 0, nu_dot[3], nu_dot[4], nu_dot[5]], float)
        

        # Forward Euler integration
        nu = nu + sampleTime * nu_dot
        n = n + sampleTime * n_dot

        u_actual = np.array(n, float)

        # return nu_dot, u_actual
        return nu_dot, u_actual
    
    def waveSpectrum_JONSWAP(self, omega):
        g = 9.81
        gamma = 3.3
        Hs = self.H_wave
        Tp = self.T_wave
        wp = 2 * np.pi / Tp

        alpha = 0.076 * (Hs**2) / (Tp**4)
        sigma = np.where(omega < wp, 0.07, 0.09)
        r = np.exp(-(omega - wp)**2 / (2 * sigma**2 * wp**2))
        return alpha * g**2 * omega**-5 * np.exp(-1.25 * (wp / omega)**4) * gamma**r
    def computeWaveForce_freq(self, t):
        tau_wave = np.zeros(6)
        rao_ratio = {
        'surge': 1.0,   # F_x
        'sway':  1.0,   # F_y
        'heave': 1.0,   # F_z
        'roll':  1.0,   # M_x
        'pitch': 1.0,   # M_y
        'yaw':   1.0    # M_z
        }
        for i in range(self.N_wave_freq):
            omega_i = self.omega_list[i]
            phase_i = self.phase_list[i]
            S_i = self.waveSpectrum_JONSWAP(omega_i)
            
            # 振幅由谱密度决定：A_i = sqrt(2 * S * delta_omega)
            delta_omega = self.omega_list[1] - self.omega_list[0]
            A_i = np.sqrt(2 * S_i * delta_omega)

            # 假设结构RAO是1，简化响应幅为 A_i
            Fx_i = A_i * rao_ratio['surge'] * np.cos(omega_i * t + phase_i)
            Fy_i = A_i * rao_ratio['sway']  * np.sin(omega_i * t + phase_i)
            Fz_i = A_i * rao_ratio['heave'] * np.cos(omega_i * t + phase_i + np.pi / 4)  # 加一点相位差

            Mx_i = A_i * rao_ratio['roll']  * np.sin(omega_i * t + phase_i + np.pi / 6)
            My_i = A_i * rao_ratio['pitch'] * np.cos(omega_i * t + phase_i + np.pi / 3)
            Mz_i = A_i * rao_ratio['yaw']   * np.sin(omega_i * t + phase_i)

            tau_wave += np.array([Fx_i, Fy_i, Fz_i, Mx_i, My_i, Mz_i])

        return tau_wave




    def controlAllocation(self, tau3):
        """
        u_alloc  = controlAllocation(tau3),  tau3 = [tau_X, tau_Y, tau_N]'
        u_alloc = B' * inv( B * B' ) * tau3
        """
        B_pseudoInv = self.B.T @ np.linalg.inv(self.B @ self.B.T)
        u_alloc = np.matmul(B_pseudoInv, tau3)

        return u_alloc


    def DPcontrol(self, eta, nu, sampleTime):
        """
        u = DPcontrol(eta,nu,sampleTime) is a nonlinear PID controller
        for DP based on pole placement:

        tau = -R' Kp (eta-r) - R' Kd R nu - R' Ki int(eta-r)
        u = B_pseudoinverse * tau
        """
        # 3-DOF state vectors
        eta3 = np.array([eta[0], eta[1], eta[5]])
        nu3 = np.array([nu[0], nu[1], nu[5]])

        # 3-DOF diagonal model matrices
        M3 = np.diag([self.M[0][0], self.M[1][1], self.M[5][5]])
        D3 = np.diag([self.D[0][0], self.D[1][1], self.D[5][5]])

        [tau3, self.e_int, self.x_d, self.y_d, self.psi_d] = DPpolePlacement(
            self.e_int,
            M3,
            D3,
            eta3,
            nu3,
            self.x_d,
            self.y_d,
            self.psi_d,
            self.wn,
            self.zeta,
            self.ref,
            sampleTime,
        )

        u_alloc = self.controlAllocation(tau3)

        # u_alloc = abs(n) * n --> n = sign(u_alloc) * sqrt(u_alloc)
        n = np.zeros(self.dimU)
        for i in range(0, self.dimU):
            n[i] = np.sign(u_alloc[i]) * math.sqrt(abs(u_alloc[i]))

        u_control = n

        return u_control

    def stepInput(self, t):
        """
        u = stepInput(t) generates propeller step inputs.
        """
        tau3 = np.array([10000, 0, 100000], float)

        if t > 30:
            tau3 = np.array([1000, 1000, 0], float)
        if t > 70:
            tau3 = np.array([0, 0, 0], float)

        # GMY
        tau3 = np.array([0, 0, 0], float)

        u_control = self.controlAllocation(tau3)

        return u_control
    
def DPpolePlacement(
    e_int, M3, D3, eta3, nu3, x_d, y_d, psi_d, wn, zeta, eta_ref, sampleTime
):

    # PID gains based on pole placement
    M3_diag = np.diag(np.diag(M3))
    D3_diag = np.diag(np.diag(D3))
    
    # 原始DP定义
    Kp = wn @ wn @ M3_diag
    Kd = 2.0 * zeta @ wn @ M3_diag - D3_diag
    Ki = (1.0 / 10.0) * wn @ Kp

    # # supply 原始DP参数
    # Kp = np.diag([6.7644e+04, 1.13412e+05, 1.78095128e+08])
    # Kd = np.diag([1.27580895e+06, 2.01356107e+06, 1.39594401e+09])
    # Ki = np.diag([676.44, 1134.12, 3561902.55])

    # Kp = np.diag([1e10, 1e10, 1e7])
    # Kd = np.diag([1e10, 0, 0])
    # Ki = np.diag([1e10, 0, 0])

    # DP control law - setpoint regulation
    e = eta3 - np.array([x_d, y_d, psi_d])
    # print("!!!error!!!:", e)
    e[2] = ssa(e[2])
    R = Rzyx(0.0, 0.0, eta3[2])
    tau = (
        - np.matmul((R.T @ Kp), e)
        - np.matmul(Kd, nu3)
        - np.matmul((R.T @ Ki), e_int)
    )


    # # Low-pass filters, Euler's method
    T = 5.0 * np.array([1 / wn[0][0], 1 / wn[1][1], 1 / wn[2][2]])
    # T[2] = T[2]/ 100
    x_d += sampleTime * (eta_ref[0] - x_d) / T[0]
    y_d += sampleTime * (eta_ref[1] - y_d) / T[1]
    psi_d += sampleTime * (eta_ref[2] - psi_d) / T[2]

    # Integral error, Euler's method
    e_int += sampleTime * e 

    return tau, e_int, x_d, y_d, psi_d
def Rzyx(phi,theta,psi):
    """
    R = Rzyx(phi,theta,psi) computes the Euler angle rotation matrix R in SO(3)
    using the zyx convention
    """
    
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cth  = math.cos(theta)
    sth  = math.sin(theta)
    cpsi = math.cos(psi)
    spsi = math.sin(psi)
    
    R = np.array([
        [ cpsi*cth, -spsi*cphi+cpsi*sth*sphi, spsi*sphi+cpsi*cphi*sth ],
        [ spsi*cth,  cpsi*cphi+sphi*sth*spsi, -cpsi*sphi+sth*spsi*cphi ],
        [ -sth,      cth*sphi,                 cth*cphi ] ])

    return R

def ssa(angle):
    """
    angle = ssa(angle) returns the smallest-signed angle in [ -pi, pi )
    """
    angle = (angle + math.pi) % (2 * math.pi) - math.pi
        
    return angle 

def sat(x, x_min, x_max):
    """
    x = sat(x,x_min,x_max) saturates a signal x such that x_min <= x <= x_max
    """
    if x > x_max:
        x = x_max 
    elif x < x_min:
        x = x_min
        
    return x  

# =================================================supply================================================================
class supply:
    """
    supply()                                      Propeller step inputs
    supply('DPcontrol',x_d,y_d,psi_d,V_c,beta_c)  DP control system
    
    Inputs:
        x_d: desired x position (m)
        y_d: desired y position (m)
        psi_d: desired yaw angle (deg)
        V_c: current speed (m/s)
        beta_c: current direction (deg)
    """

    def __init__(
        self,
        controlSystem="stepInput",
        r_x = 0.0,
        r_y = 0.0,
        r_n = 0.0,
        V_current = 0.0,
        beta_current = 0.0,
    ):
        
        # Constants
        D2R = math.pi / 180     # deg2rad
        g = 9.81                # acceleration of gravity (m/s^2)

        if controlSystem == "DPcontrol":
            self.controlDescription = (
                "Nonlinear DP control (x_d, y_d, psi_d) = ("
                + str(r_x)
                + " m, "
                + str(r_y)
                + " m, "
                + str(r_n)
                + " deg)"
            )

        else:
            self.controlDescription = "Step inputs n = [n1, n2, n3, n4, n5, n6]"
            controlSystem = "stepInput"

        self.ref = np.array([r_x, r_y, r_n * D2R], float)
        self.V_c = V_current
        self.beta_c = beta_current * D2R
        self.controlMode = controlSystem

        # Initialize the supply vessel model
        m = 6000.0e3        # mass (kg)
        self.L = 76.2       # length (m)
        self.T_n = 1.0      # prop. speed time constant (s)
        self.n_max = np.array([250, 250, 250, 250, 
                               160, 160], float) # RPM saturation limits
        self.nu = np.array([0, 0, 0, 0, 0, 0], float) # initial velocity vector
        self.u_actual = np.array([0, 0, 0, 0, 0, 0], float) # RPM inputs
        self.name = "Offshore supply vessel (see 'supply.py' for more details)"

        # Two tunnel thrusters in the bow, no. 1 and 2
        # Two tunnel thrusters in the stern, no. 3 and 4
        # Two main propellers aft, no. 3 and 4
        self.controls = [
            "#1 Bow tunnel thruster (RPM)",
            "#2 Bow tunnel thruster (RPM)",
            "#3 Stern tunnel thruster (RPM)",
            "#4 Stern tunnel thruster (RPM)",
            "#5 Right main propeller (RPM)",
            "#6 Left main propeller (RPM)"
        ]
        self.dimU = len(self.controls)

        # Thrust coefficient and configuration matrices (Fossen 2021, Ch. 11.2)
        # Thrust_max(i) = K(i) * n_max(i)^2
        # Tunnel thruster: 3.2 * 250^2 = 200 kN
        # Main propeller: 31.2 * 160^2 = 799 kN
        K = np.diag([3.2, 3.2, 3.2, 3.2, 31.2, 31.2])
        T = np.array(
            [ [0, 0, 0, 0, 1, 1], [1, 1, 1, 1, 0, 0], 
              [30, 22, -22, -30, -8, 8] ], float
        )
        self.B = T @ K

        # Tbis = np.diag( [1, 1, 1 / self.L],float)
        Tbis_inv = np.diag([1.0, 1.0, self.L])

        # 3-DOF model matrices - bis scaling (Fossen 2021, App. D)
        Mbis = np.array(
            [[1.1274, 0, 0], [0, 1.8902, -0.0744], [0, -0.0744, 0.1278]], float
        )

        Dbis = np.array(
            [[0.0358, 0, 0], [0, 0.1183, -0.0124], [0, -0.0041, 0.0308]], float
        )

        self.M3 = m * Tbis_inv @ Mbis @ Tbis_inv
        self.M3inv = np.linalg.inv(self.M3)
        self.D3 = m * math.sqrt(g / self.L) * Tbis_inv @ Dbis @ Tbis_inv

        # DP control system
        self.e_int = np.array([0, 0, 0], float)  # integral states
        self.x_d = 0.0  # setpoints
        self.y_d = 0.0
        self.psi_d = 0.0
        self.wn = np.diag([0.1, 0.1, 0.2])    # PID pole placement
        self.zeta = np.diag([1.0, 1.0, 1.0])


    def dynamics(self, eta, nu, u_actual, u_control, sampleTime):
        """
        [nu,u_actual] = dynamics(eta,nu,u_actual,u_control,sampleTime) integrates the
        supply vessel equations of motion using Euler's method.
        """

        # Input vector
        n = u_actual  # propeller speed (RPM)

        # # Current velocities
        u_c = self.V_c * math.cos(self.beta_c - eta[3])  # current surge velocity   eta[5]
        v_c = self.V_c * math.sin(self.beta_c - eta[4])  # current sway velocity    eta[5]
        w_c = self.V_c * math.sin(eta[5])  # current yaw velocity
        x_c = self.V_c * math.sin(eta[0] - self.beta_c)  # current x velocity
        y_c = self.V_c * math.cos(eta[1] - self.beta_c)  # current y velocity
        z_c = self.V_c * math.cos(eta[2])  # current z velocity


        # GMY
        # # 修改海浪速度计算方式，使其与平台姿态成负相关
        # # 使用姿态角eta[3]和eta[4]作为输入，计算恢复力方向
        # u_c = -self.V_c * eta[3] * 0.5 # eta[3]正向过大时，u_c为负，产生恢复力
        # v_c = -self.V_c * eta[4] * 0.5 # eta[4]正向过大时，v_c为负，产生恢复力
        
        # if self.wave == 0:
        #     u_c += 0.1
        #     v_c += 0.1
        #     self.wave += 1
        # else:
        #     u_c += np.random.uniform(-0.05, 0.05) / self.wave * (abs(eta[3]) <= 0.001)
        #     v_c += np.random.uniform(-0.05, 0.05) / self.wave * (abs(eta[3]) <= 0.001)
        #     self.wave += 1
            
            
        # 限制最大海浪速度
        # max_wave_speed = 0.5  # 设置最大海浪速度
        # u_c = np.clip(u_c, -max_wave_speed, max_wave_speed)
        # v_c = np.clip(v_c, -max_wave_speed, max_wave_speed)

        nu_c = np.array([x_c, y_c, z_c, u_c, v_c, w_c], float)  # current velocity vector​
        nu_r = nu - nu_c  # relative velocity vector

        # Control forces and moments with propeller saturation
        n_squared = np.zeros(self.dimU)
        for i in range(0, self.dimU):
            n[i] = sat(n[i], -self.n_max[i], self.n_max[i])      # saturation
            n_squared[i] = abs(n[i]) * n[i]

        tau3 = np.matmul(self.B, n_squared)

        # 3-DOF dynamics
        nu3_r = np.array([nu_r[0], nu_r[1], nu_r[5]])
        nu3_dot = np.matmul(self.M3inv, tau3 - np.matmul(self.D3, nu3_r))

        # 6-DOF ship model and propeller speed dynamics
        nu_dot = np.array([nu3_dot[0], nu3_dot[1], 0, 0, 0, nu3_dot[2]])
        n_dot = (u_control - u_actual) / self.T_n

        # Forward Euler integration
        nu = nu + sampleTime * nu_dot
        n = n + sampleTime * n_dot

        u_actual = np.array(n, float)

        return nu_dot, u_actual

    def controlAllocation(self, tau3):
        """
        u_alloc  = controlAllocation(tau3),  tau3 = [tau_X, tau_Y, tau_N]'
        u_alloc = B' * inv( B * B' ) * tau3
        """
        B_pseudoInv = self.B.T @ np.linalg.inv(self.B @ self.B.T)
        u_alloc = np.matmul(B_pseudoInv, tau3)  # squared propeller speed

        return u_alloc


    def DPcontrol(self, eta, nu, sampleTime):
        """
        u = DPcontrol(eta,nu,sampleTime) is a nonlinear PID controller
        for DP based on pole placement:

        tau = -R' Kp (eta-r) - Kd nu - R' Ki int(eta-r)
        u = B_pseudoinverse * tau
        """
        eta3 = np.array([eta[0], eta[1], eta[5]])
        nu3 = np.array([nu[0], nu[1], nu[5]])

        [tau3, self.e_int, self.x_d, self.y_d, self.psi_d] = DPpolePlacement(
            self.e_int,
            self.M3,
            self.D3,
            eta3,
            nu3,
            self.x_d,
            self.y_d,
            self.psi_d,
            self.wn,
            self.zeta,
            self.ref,
            sampleTime,
        )

        u_alloc = self.controlAllocation(tau3)

        # u_alloc = abs(n) * n --> n = sign(u_alloc) * sqrt(u_alloc)
        n = np.zeros(self.dimU)
        for i in range(0, self.dimU):
            n[i] = np.sign(u_alloc[i]) * math.sqrt(abs(u_alloc[i]))

        u_control = n

        return u_control


    def stepInput(self, t):
        """
        u = stepInput(t) generates propeller step inputs (RPM).
        """
        n = np.array([0, 0, 0, 0, 100, 100], float)

        if t > 30:
            n = np.array([50, 50, 50, 50, 50, 50], float)
        if t > 70:
            n = np.array([0, 0, 0, 0, 0, 0], float)

        # GMY
        n = np.array([0, 0, 0, 0, 100, 100], float)
        u_control = n

        return u_control