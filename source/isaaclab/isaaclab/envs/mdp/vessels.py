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
    


# ===========================================================  MSS ===========================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy import io
import time
from typing import Tuple, Dict, Any
import os
from scipy.interpolate import interp1d

class VesselControlSystem:
    def __init__(self, target_position=None, initial_eta=None, initial_nu=None, dt=0.02):
        """
        初始化船舶控制系统
        
        参数:
            target_position: 期望位置 [x, y, yaw]，默认为 [10, 10, π]
            initial_eta: 初始位置 [x, y, z, roll, pitch, yaw]，默认为 [0, 0, 0, 10°, 0, 0]
            initial_nu: 初始速度 [u, v, w, p, q, r]，默认为 [0, 0, 0, 0, 0, 0]
        """
        self.dt = dt  # 调整为更小的时间步长，适合IsaacLab环境
        self.eta_r_ddot = np.zeros(3)
        self.omega_o = 0.8976 * np.array([0.08, 0.1, 0.1])  # 保守的收敛速度设置
        self.omega_c = 1.2255 * self.omega_o
        self.DELTA = np.diag([1, 1, 1])
        
        # 更保守的控制参数 - 防止振荡
        self.Kp = np.diag([5e7, 5e7, 2e10])  # 进一步降低比例增益 1e5 * np.diag([5e2, 5e2, 2e5])
        self.Kd = np.diag([2e4, 2e4, 2e6])  # 增加阻尼，抑制振荡 1e4 * np.diag([2e1, 2e1, 2e2]) 
        self._controller_gains = True
        
        # 禁用自适应控制（使用原始设置）
        self._adaptive_control = False
        self._max_force_threshold = 1e6   # 原始控制力限制
        self._adaptation_rate = 0.1       # 原始自适应调整速率
        self._error_threshold = 1.0       # 原始误差阈值
        self._min_gain_ratio = 0.1        # 原始最小增益比例
        
        # 噪声抑制参数
        self._noise_filter_alpha = 0.8     # 低通滤波器系数
        self._measurement_noise_std = 0.01 # 测量噪声标准差
        self._control_saturation = 1e6     # 原始控制力饱和限制 # GMY 1e6是能跑的
        
        # 鲁棒观测器参数
        self._observer_robustness = True
        self._observer_noise_gain = 0.1   # 观测器噪声增益
        self._observer_damping = 0.9      # 观测器阻尼系数
        
        # 初始化滤波器状态
        self._filtered_eta = np.zeros(6)
        self._filtered_nu = np.zeros(6)
        self._filtered_error = np.zeros(3)
        
        # 预初始化观测器增益（避免属性错误）
        self.K4 = 1e3 * np.diag([0.1, 0.1, 0.001])
        
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
            vessel_data = io.loadmat('/home/user/IsaacLab/source/isaaclab/isaaclab/envs/mdp/vessel.mat')
            vesselABC_data = io.loadmat('/home/user/IsaacLab/source/isaaclab/isaaclab/envs/mdp/vesselABC.mat')

            self.vessel = vessel_data['vessel'][0, 0] if 'vessel' in vessel_data else (print("vessel数据未找到") or {})
            self.vesselABC = vesselABC_data['vesselABC'][0, 0] if 'vesselABC' in vesselABC_data else (print("vesselABC数据未找到") or {})
            self.inv_M = self.vesselABC['Minv'] if 'Minv' in self.vesselABC.dtype.names else (print("Minv数据未找到") or np.eye(6))
            self.G = self.vesselABC['G'] if 'G' in self.vesselABC.dtype.names else (print("G数据未找到") or np.zeros((6, 6)))
            self.D = self.vessel['Bv'][:, :, 0] if 'Bv' in self.vessel.dtype.names else (print("Bv数据未找到") or np.zeros((6, 6)))
            self.C = np.zeros((6, 6))
            
            # 确保质量矩阵被正确设置
            if not hasattr(self, 'M'):
                self.M = np.linalg.inv(self.inv_M)
        except FileNotFoundError:
            print("警告: 未找到船舶数据文件，使用默认参数")
            self.vessel = type('obj', (object,), {})()
            self.vesselABC = type('obj', (object,), {})()
            # 使用合理的默认质量矩阵
            self.M = np.diag([1000, 1000, 1000, 100, 100, 100])  # 合理的质量矩阵
            self.inv_M = np.linalg.inv(self.M)
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

    def calculate_memory_effects(self, nu_r) -> np.ndarray:
        if self.memory_systems is None:
            print("没有memory_systems")
            return np.zeros(6)
            
        # 处理输入数据类型
        if hasattr(nu_r, 'cpu'):  # 如果是torch张量
            nu_r_np = nu_r.detach().cpu().numpy()
        else:  # 如果是numpy数组
            nu_r_np = nu_r
            
        nu_components = [nu_r_np[0], nu_r_np[2], nu_r_np[4],  # u, w, q
                        nu_r_np[1], nu_r_np[3], nu_r_np[5],  # v, p, r
                        nu_r_np[0], nu_r_np[2], nu_r_np[4],  # u, w, q
                        nu_r_np[1], nu_r_np[3], nu_r_np[5],  # v, p, r
                        nu_r_np[0], nu_r_np[2], nu_r_np[4],  # u, w, q
                        nu_r_np[1], nu_r_np[3], nu_r_np[5]]  # v, p, r
        
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
        self.b_hat = 0 * x_next[9:12]
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
            self.K4 = 1e3 * np.diag([0.1, 0.1, 0.001])  # 原始观测器增益
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
        
        # 简单的观测器更新
        y_tilde = y - y_hat
        R = self.Rzyx(np.array([0, 0, y[2]]))
        
        # 修正：确保维度匹配，并添加鲁棒性
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
    
    def controller_acceleration(self, eta_r, x_hat, b_hat, 
                               current_eta, current_nu):
        """控制器 - 返回控制加速度nu_dot"""
        # 处理输入数据类型
        if hasattr(eta_r, 'cpu'):  # 如果是torch张量
            eta_r_np = eta_r.detach().cpu().numpy()
            x_hat_np = x_hat.detach().cpu().numpy()
            b_hat_np = b_hat.detach().cpu().numpy()
            current_eta_np = current_eta.detach().cpu().numpy()
            current_nu_np = current_nu.detach().cpu().numpy()
            is_tensor = True
        else:  # 如果是numpy数组
            eta_r_np = eta_r
            x_hat_np = x_hat
            b_hat_np = b_hat
            current_eta_np = current_eta
            current_nu_np = current_nu
            is_tensor = False
            
        # 控制器参数已在__init__中设置，这里不需要重复设置
            
        eta_hat, nu_hat = x_hat_np[0:3], x_hat_np[3:6]
        error = eta_hat - eta_r_np
        R = self.Rzyx(np.array([0, 0, eta_hat[2]]))
        
        # 计算控制力（保持原始符号）
        u = -R.T @ (self.Kp @ error + self.Kd @ R @ nu_hat + b_hat_np)
        
        # 分维度检测是否达到目标位置附近，一旦达到就持续限制（这是没有办法的办法）
        # if not hasattr(self, '_target_reached_flags'):
        #     self._target_reached_flags = [False, False, False]  # [X, Y, Z]方向标志
        #     self.xianzhi1 = 1e6
        #     self.xianzhi2 = 1e6
        #     self.xianzhi3 = 1e8
        # # 分维度检查是否达到目标位置附近
        # if abs(error[0]) < 0.1:  # X方向接近目标
        #     self._target_reached_flags[0] = True
        # if abs(error[1]) < 0.1:  # Y方向接近目标
        #     self._target_reached_flags[1] = True
        # if abs(error[2]) < 0.005:  # Z方向接近目标
        #     self._target_reached_flags[2] = True
        # # 根据各维度是否达到目标来限制控制力
        # if self._target_reached_flags[0]:  # X方向已达到目标
        #     u[0] = np.clip(u[0], -self.xianzhi1, self.xianzhi1)   # X方向严格限制
        # else:
        #     pass
            
        # if self._target_reached_flags[1]:  # Y方向已达到目标
        #     u[1] = np.clip(u[1], -self.xianzhi1, self.xianzhi1)   # Y方向严格限制
        # else:
        #     pass
            
        # if self._target_reached_flags[2]:  # Z方向已达到目标
        #     u[2] = np.clip(u[2], -self.xianzhi3, self.xianzhi3)   # Z方向严格限制
        # else:
        #     pass
        
        # 存储控制力到系统状态中
        self.u = u
        
        # 将控制力转换为6DOF推力
        tau_thruster = np.array([u[0], u[1], 0, 0, 0, u[2]])
        
        # 计算其他力
        nu_r = current_nu_np
        tau_cf = self.crossflow_drag(nu_r)
        mef = self.calculate_memory_effects(nu_r)
        damping_force = self.D @ nu_r
        gravity_force = self.G @ current_eta_np
        
        # 计算总加速度
        nu_dot = self.inv_M @ (tau_thruster - self.C @ nu_r - damping_force - 
                             gravity_force + tau_cf - mef)
        
        return nu_dot

    def crossflow_drag(self, nu_r) -> np.ndarray:
        """优化的横流阻力计算"""
        # 处理输入数据类型
        if hasattr(nu_r, 'cpu'):  # 如果是torch张量
            nu_r_np = nu_r.detach().cpu().numpy()
        else:  # 如果是numpy数组
            nu_r_np = nu_r
            
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
            
        u_r, v_r, r = nu_r_np[0], nu_r_np[1], nu_r_np[5]
        
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

    def step(self, current_eta, current_nu, current_time: float):
        """
        单步计算函数
        
        输入:
            current_eta: 当前时刻的位置 [x, y, z, roll, pitch, yaw] (可以是numpy数组或torch张量)
            current_nu: 当前时刻的速度 [u, v, w, p, q, r] (可以是numpy数组或torch张量)
            current_time: 当前时间
            
        输出:
            next_eta: 下一个时刻的位置
            next_nu: 下一个时刻的速度  
            current_control_acceleration: 当前时刻的控制加速度nu_dot
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
            
        # 更新内部状态（直接使用IsaacLab的状态）
        self.eta = current_eta_np.copy()
        self.nu = current_nu_np.copy()
        
        # 生成测量噪声（降低噪声水平）
        y = self.eta[[0, 1, 5]] 
        
        # 参考轨迹计算
        eta_r, eta_r_dot = self.reference[0:3], self.reference[3:6]
        self.eta_r_ddot = self.reference_model_dynamics(eta_r, eta_r_dot)
        reference_dot = np.concatenate([eta_r_dot, self.eta_r_ddot])
        
        # 使用观测器（恢复完整的观测器功能）
        y_hat = self.x_hat[0:3]
        # self.x_hat, self.b_hat, self.xi_hat, self.nu_hat = self.observer_dynamics(self.u, y, y_hat)
        self.x_hat = np.concatenate([self.eta[[0, 1, 5]], self.nu[[0, 1, 5]]])
        self.nu_hat = self.nu[[0, 1, 5]]
        self.b_hat = 0 * self.b_hat
        self.xi_hat = 0 * self.xi_hat
        
        # 控制器计算 - 使用完整的控制器
        current_control_acceleration = self.controller_acceleration(eta_r, self.x_hat, self.b_hat, 
                                                                   current_eta_np, current_nu_np)
        
        # 添加波浪载荷（允许平台在波浪中运动）
        wave_loads = self.generate_wave_loads_jonswap(current_time)
        current_control_acceleration += self.inv_M @ wave_loads
        
        # 计算位置导数
        R = self.Rzyx(self.eta[3:6])
        T_mat = self.T_Theta(self.eta[3:6])
        eta_dot = np.concatenate([R @ self.nu[:3], T_mat @ self.nu[3:6]])
        
        # 内部状态更新（用于控制器计算，但不影响IsaacLab物理引擎）
        self.eta = self.eta + eta_dot * self.dt
        self.nu = self.nu + current_control_acceleration * self.dt
        
        # 更新参考轨迹
        self.reference += reference_dot * self.dt
        
        # 根据输入类型返回相应格式的结果
        if is_tensor:
            import torch
            return torch.from_numpy(current_control_acceleration).to(
                dtype=current_eta.dtype, device=current_eta.device
            ), torch.from_numpy(eta_dot).to(
                dtype=current_eta.dtype, device=current_eta.device
            )
        else:
            return current_control_acceleration, eta_dot
    
    def generate_wave_loads_jonswap(self, t):
        """
        极致优化版波浪载荷生成函数
        保持结果完全一致，仅加速计算
        """
        if not hasattr(self, '_wave_init'):
            Hs = 3.0
            Tp = 8
            g = 9.81
            omega_p = 2 * np.pi / Tp
            gamma = 3.3
            
            # RAO权重系数 - 可以调节每个自由度的波浪响应强度
            # [Surge, Sway, Heave, Roll, Pitch, Yaw]
            self._rao_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            
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
            
            # 合并两次求和为 einsum（最优），并应用RAO权重
            tau_wave[d] = self._rao_weights[d] * np.einsum('i,ij,ij->', weight, amp_d, cos_val)

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