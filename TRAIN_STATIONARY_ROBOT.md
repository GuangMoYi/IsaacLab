# 训练相对平台静止的机器狗 - 实现方案

## 目标
训练一个机器狗，使其能够保持相对于运动平台的静止（即机器狗在平台上的位置和姿态保持不变）。

## 实现方案

### 方案1：修改命令生成器（最简单）

**原理**：将目标速度命令设置为零，让机器狗学习保持静止。

**修改文件**：`source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`

```python
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            # 将目标速度设置为零（相对于平台）
            lin_vel_x=(0.0, 0.0),      # 改为 (0.0, 0.0)
            lin_vel_y=(0.0, 0.0),      # 改为 (0.0, 0.0)
            ang_vel_z=(0.0, 0.0),      # 改为 (0.0, 0.0)
            heading=(-math.pi, math.pi)  # 可以保持，或者也设为 (0.0, 0.0)
        ),
    )
```

**优点**：
- 实现简单，只需修改一行配置
- 不需要修改观测空间和奖励函数
- 机器狗会自动学习保持静止

**缺点**：
- 机器狗可能无法感知平台运动，导致在平台运动时无法保持相对静止
- 需要平台运动较慢或机器狗有足够的反应时间

---

### 方案2：添加平台速度观测 + 零速度命令（推荐）

**原理**：让机器狗感知平台的运动，同时将目标速度设置为零。

**步骤1：添加平台速度观测**

修改 `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`：

```python
@configclass
class ObservationsCfg:
    """观测量配置"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # 原有观测
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=0, n_max=0))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=0, n_max=0))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=0, n_max=0))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=0, n_max=0))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=0, n_max=0))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=0, n_max=0),
            clip=(-1.0, 1.0),
        )
        
        # ========== 新增：平台速度观测 ==========
        # 平台线速度（世界坐标系，在机器人体坐标系下）
        platform_lin_vel_rel = ObsTerm(
            func=mdp.platform_lin_vel_w, 
            noise=Unoise(n_min=0, n_max=0),
            params={"asset_cfg": SceneEntityCfg("platform")}
        )
        # 平台角速度（世界坐标系，在机器人体坐标系下）
        platform_ang_vel_rel = ObsTerm(
            func=mdp.platform_ang_vel_w,
            noise=Unoise(n_min=0, n_max=0),
            params={"asset_cfg": SceneEntityCfg("platform")}
        )
        # ========================================

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
```

**步骤2：修改命令生成器（与方案1相同）**

将目标速度设置为零。

**优点**：
- 机器狗可以感知平台运动
- 能够更好地适应平台运动
- 实现相对简单

**缺点**：
- 需要添加观测项（但已有现成的函数）

---

### 方案3：计算相对速度 + 零速度命令（最准确）

**原理**：计算机器狗相对于平台的速度，并将目标相对速度设置为零。

**步骤1：创建相对速度观测函数**

在 `source/isaaclab/isaaclab/envs/mdp/observations.py` 中添加：

```python
@generic_io_descriptor(
    units="m/s",
    axes=["X", "Y", "Z"],
    observation_type="RelativeVelocity",
    on_inspect=[record_shape, record_dtype],
)
def robot_relative_vel_to_platform(
    env: ManagerBasedEnv, 
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    platform_cfg: SceneEntityCfg = SceneEntityCfg("platform")
) -> torch.Tensor:
    """机器狗相对于平台的线速度（在机器人体坐标系下）"""
    robot: Articulation = env.scene[robot_cfg.name]
    platform: RigidObject = env.scene[platform_cfg.name]
    
    # 机器狗和平台的世界坐标系速度
    robot_vel_w = robot.data.root_lin_vel_w
    platform_vel_w = platform.data.root_lin_vel_w
    
    # 计算相对速度（世界坐标系）
    rel_vel_w = robot_vel_w - platform_vel_w
    
    # 转换到机器人体坐标系
    rel_vel_b = math_utils.quat_apply_inverse(robot.data.root_quat_w, rel_vel_w)
    
    return rel_vel_b


@generic_io_descriptor(
    units="rad/s",
    axes=["X", "Y", "Z"],
    observation_type="RelativeVelocity",
    on_inspect=[record_shape, record_dtype],
)
def robot_relative_ang_vel_to_platform(
    env: ManagerBasedEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    platform_cfg: SceneEntityCfg = SceneEntityCfg("platform")
) -> torch.Tensor:
    """机器狗相对于平台的角速度（在机器人体坐标系下）"""
    robot: Articulation = env.scene[robot_cfg.name]
    platform: RigidObject = env.scene[platform_cfg.name]
    
    # 机器狗和平台的世界坐标系角速度
    robot_ang_vel_w = robot.data.root_ang_vel_w
    platform_ang_vel_w = platform.data.root_ang_vel_w
    
    # 计算相对角速度（世界坐标系）
    rel_ang_vel_w = robot_ang_vel_w - platform_ang_vel_w
    
    # 转换到机器人体坐标系
    rel_ang_vel_b = math_utils.quat_apply_inverse(robot.data.root_quat_w, rel_ang_vel_w)
    
    return rel_ang_vel_b
```

**步骤2：在观测配置中使用相对速度**

```python
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # ... 原有观测 ...
        
        # ========== 新增：相对速度观测 ==========
        robot_relative_lin_vel = ObsTerm(
            func=mdp.robot_relative_vel_to_platform,
            noise=Unoise(n_min=0, n_max=0)
        )
        robot_relative_ang_vel = ObsTerm(
            func=mdp.robot_relative_ang_vel_to_platform,
            noise=Unoise(n_min=0, n_max=0)
        )
        # ========================================
```

**步骤3：修改命令生成器（与方案1相同）**

**优点**：
- 最准确：直接观测相对速度
- 机器狗可以精确感知相对于平台的运动
- 训练效果最好

**缺点**：
- 需要添加新的观测函数
- 实现稍复杂

---

### 方案4：添加相对位置奖励（最完整）

**原理**：不仅观测相对速度，还添加奖励项来惩罚机器狗在平台上的位置变化。

**步骤1-3**：与方案3相同

**步骤4：添加相对位置奖励函数**

在 `source/isaaclab/isaaclab/envs/mdp/rewards.py` 中添加：

```python
def relative_position_stability(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    platform_cfg: SceneEntityCfg = SceneEntityCfg("platform"),
    std: float = 0.1,
) -> torch.Tensor:
    """奖励机器狗在平台上保持相对位置稳定"""
    robot: Articulation = env.scene[robot_cfg.name]
    platform: RigidObject = env.scene[platform_cfg.name]
    
    # 初始化相对位置（第一次调用时）
    if not hasattr(env, '_initial_relative_pos'):
        robot_pos_w = robot.data.root_pos_w
        platform_pos_w = platform.data.root_pos_w
        # 计算初始相对位置（在平台坐标系下）
        rel_pos_w = robot_pos_w - platform_pos_w
        platform_quat_inv = math_utils.quat_conjugate(platform.data.root_quat_w)
        rel_pos_p = math_utils.quat_apply(platform_quat_inv, rel_pos_w)
        env._initial_relative_pos = rel_pos_p.clone()
    
    # 计算当前相对位置
    robot_pos_w = robot.data.root_pos_w
    platform_pos_w = platform.data.root_pos_w
    rel_pos_w = robot_pos_w - platform_pos_w
    platform_quat_inv = math_utils.quat_conjugate(platform.data.root_quat_w)
    rel_pos_p = math_utils.quat_apply(platform_quat_inv, rel_pos_w)
    
    # 计算位置误差（只考虑xy平面）
    pos_error = torch.linalg.norm(
        (rel_pos_p[:, :2] - env._initial_relative_pos[:, :2]), dim=1
    )
    
    # 使用指数奖励
    return torch.exp(-pos_error / std)
```

**步骤5：在奖励配置中添加**

```python
@configclass
class RewardsCfg:
    # ... 原有奖励 ...
    
    # ========== 新增：相对位置稳定性奖励 ==========
    relative_position_stability = RewTerm(
        func=mdp.relative_position_stability,
        weight=2.0,  # 可以根据需要调整权重
        params={"std": 0.1}
    )
    # ==============================================
```

**优点**：
- 最完整：同时考虑速度和位置
- 训练效果最好
- 机器狗能够精确保持相对静止

**缺点**：
- 实现最复杂
- 需要添加多个函数

---

## 推荐方案

**对于快速测试**：使用**方案1**（最简单）

**对于实际应用**：使用**方案2**或**方案3**（平衡了复杂度和效果）

**对于最佳效果**：使用**方案4**（最完整）

## 实施建议

1. **先尝试方案1**：如果平台运动较慢，可能已经足够
2. **如果效果不好，升级到方案2**：添加平台速度观测
3. **如果需要更精确的控制，使用方案3或4**：计算相对速度和位置

## 注意事项

1. **平台运动速度**：如果平台运动太快，机器狗可能无法及时反应
2. **奖励权重**：需要仔细调整奖励权重，确保机器狗能够学习保持静止
3. **训练时间**：相对静止任务可能需要更长的训练时间
4. **平台运动模式**：确保平台运动模式多样化，以提高泛化能力

