# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """用于配置一个有地形和四足机器人的仿真场景"""

    # 地形配置
        # 指定地形在仿真场景中的路径
        # 地形类型为"generator"，表示使用地形生成器来生成地形
        # 地形生成器配置为ROUGH_TERRAINS_CFG，这是一个预定义的地形生成器配置
        # 最大初始化地形级别为5，这意味着地形将在初始化时生成5个级别
        # 碰撞组设置为-1，表示地形将与所有其他碰撞组交互
        # 物理材料配置为RigidBodyMaterialCfg，用于定义地形的物理属性
            # 摩擦力组合模式为"multiply"，表示摩擦力将在多个碰撞体之间相乘
            # 恢复系数组合模式为"multiply"，表示恢复系数将在多个碰撞体之间相乘
            # 静摩擦力为1.0
            # 动摩擦力为1.0
        # 视觉材料配置为MdlFileCfg，用于定义地形的视觉属性
            # 指定MDL文件的路径，这里使用了一个预定义的地形材料 
            # 启用UVW投影，这将使地形的纹理在三维空间中正确映射
            # 纹理缩放为(0.25, 0.25)，这将使地形的纹理缩小到原来的1/4大小
        # debug_vis设置为False，表示不启用地形的调试可视化
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            # GMY地面摩擦力，正常都是1.0
            static_friction=0.0,
            dynamic_friction=0.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # 机器人配置
    robot: ArticulationCfg = MISSING
    # 高度传感器配置
        # 附着在机器人的base上
        # 偏移量为(0.0, 0.0, 20.0)，这意味着传感器将在机器人的z轴方向上偏移20米
        # 只附着yaw轴，这意味着传感器将只关注机器人的旋转
        # 模式配置为GridPatternCfg，这是一个网格模式，用于生成高度扫描数据
            # 分辨率为0.1，这意味着每个网格点之间的距离为0.1米
            # 大小为[1.6, 1.0]，这意味着扫描区域的宽度为1.6米，高度为1.0米
        # debug_vis设置为False，表示不启用高度传感器的调试可视化
        # mesh_prim_paths设置为["/World/ground"]，这意味着高度传感器将只关注地面
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # 接触传感器配置
        # 附着在机器人的所有关节上，判断是否与地面接触，只记录3次接触数据
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # 创建天空照明
        # 资产路径
        # 光照强度为750.0， 中等偏亮的值，模拟阳光明媚的户外环境
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    
    # # GMY
    from isaaclab.assets import RigidObject, RigidObjectCfg
    # GMY 注： 这里的size是立方体的形状，而pos指的是中心的位置，机器人的初始位置要 ＞ 2 * pos_z
    # GMY 注： 这里的pos最好大于1/2 * size_z， 以防止平台与地面碰撞
    # GMY 注： 刚体默认是具有线阻尼和角阻尼的， 这里可以设置为0
    # GMY 注： 只有在刚体为正方体（三轴尺寸一致时），系统的角速度才不会随意变化，即避免Euler instability （只有在速度较大时才会有明显影响）
    platform = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Platform",
        spawn=sim_utils.CuboidCfg(
            size=(100, 100, 50),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,                                                                  # 受重力影响(False)
                enable_gyroscopic_forces=True,                                                          # 允许自由旋转
                linear_damping=0.0,
                angular_damping=0.0,
                # kinematic_enabled = True,                                                             # 运动学物体，不受力
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1e10),                                        # 质量
            collision_props=sim_utils.CollisionPropertiesCfg(
                # collision_enabled=False,       # 保持碰撞启用
                # contact_offset=0.0,           # 接触偏移设为0（无预接触力）
                # rest_offset=0.0,              # 静止偏移设为0（无静止时的微小力）
                # torsional_patch_radius=0.0,
            ),               # 碰撞属性 
            # collision_props = None,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0, 0), metallic=0.2),       # 颜色
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0),                       # 摩擦力 1.0 
            semantic_tags=[("class", "platform")],                                                      # 语义标签
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
        pos=[0, 0, 100.0],                                                                              # 初始位置
        rot=[1, 0, 0, 0],                                                                               # 初始姿态(四元数)                                                                           
        ),
    )






##
# MDP settings
##


@configclass
class CommandsCfg:
    """控制总体速度和朝向的命令配置"""

    """ 均匀采样速度命令： 应用于robot; 
        每10s重新设置一次速度
        在站立位置的2%范围内采样速度
        在朝向位置的100%范围内采样速度
        启用朝向控制
        朝向控制的刚度为0.5,值越高机器人越强硬地对齐朝向
        启用调试可视化
        速度范围是x:(-1.0, 1.0), y:(-1.0, 1.0), z:(-1.0, 1.0), heading:(-π, π)"""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class ActionsCfg:
    """控制关节动作的配置"""
    """策略网络输出一个在 [-1, 1] 区间的动作向量， 经 scale=0.5 缩放后变为 ±0.5 rad 的角度偏移并作为机器人的关节控制目标"""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


# @configclass
# class ObservationsCfg:
#     """观测量配置"""

#     @configclass
#     class PolicyCfg(ObsGroup):
#         """Observations for policy group."""

#         # 增加观测噪声
#         base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
#         base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
#         projected_gravity = ObsTerm(
#             func=mdp.projected_gravity,
#             noise=Unoise(n_min=-0.05, n_max=0.05),
#         )
#         velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
#         joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
#         joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
#         actions = ObsTerm(func=mdp.last_action)
#         # 地形高度扫描
#         height_scan = ObsTerm(
#             func=mdp.height_scan,
#             params={"sensor_cfg": SceneEntityCfg("height_scanner")},
#             noise=Unoise(n_min=-0.1, n_max=0.1),
#             clip=(-1.0, 1.0),
#         )

#         # 允许注入观测扰动（比如噪声、传感器掉线等鲁棒训练用手段）
#         def __post_init__(self):
#             self.enable_corruption = True
#             self.concatenate_terms = True

#     # 将上述量给策略网络，作为输入
#     policy: PolicyCfg = PolicyCfg()

# GMY
@configclass
class ObservationsCfg:
    """观测量配置"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # 增加观测噪声
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=0, n_max=0))   # (n_min=-0.1, n_max=0.1)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=0, n_max=0))   # (n_min=-0.2, n_max=0.2)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=0, n_max=0),                                             # (n_min=-0.05, n_max=0.05
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=0, n_max=0))     # (n_min=-0.01, n_max=0.01)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=0, n_max=0))     # (n_min=-1.5, n_max=1.5)
        actions = ObsTerm(func=mdp.last_action)
        # 地形高度扫描，clip设置裁剪后的范围
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=0, n_max=0),                                             # (n_min=-0.1, n_max=0.1)
            clip=(-1.0, 1.0),
        )


        # 允许注入观测扰动（比如噪声、传感器掉线等鲁棒训练用手段）
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    @configclass
    class DebugCfg(ObsGroup):
        platform_ang_acc = ObsTerm(func=mdp.platform_ang_acc_w)
        platform_lin_acc = ObsTerm(func=mdp.platform_lin_acc_w)
        platform_ang_vel = ObsTerm(func=mdp.platform_ang_vel_w)
        platform_lin_vel = ObsTerm(func=mdp.platform_lin_vel_w)
        platform_ang_w = ObsTerm(func=mdp.platform_ang_w)
        platform_pos_w = ObsTerm(func=mdp.platform_pos_w)
        robot_ang_acc_w = ObsTerm(func=mdp.robot_ang_acc_w)
        robot_lin_acc_w = ObsTerm(func=mdp.robot_lin_acc_w)

        def __post_init__(self):
            # 这一组不拼接为策略输入，只用于可视化或 log
            self.enable_corruption = False
            self.concatenate_terms = False

    # 将上述量给策略网络，作为输入
    policy: PolicyCfg = PolicyCfg()

    # GMY
    debug: DebugCfg = DebugCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # # interval
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )


    # # # GMY（训练测试要改）
    # # 给机器人添加一个事件，线速度角速度
    # 注：一些事件的命名可能导致代码无法运行，如：命名为push_robot，事件不执行
    push_velocity = EventTerm(
        func=mdp.move_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),                         # 每1.5~2.5秒触发一次，更加接近漂浮节奏
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {
                # "yaw": (0.5, 0.5),                        # 角速度减小，避免旋转太快
                # "roll": (0.0, 0.0), 
                # "pitch": (0, 0), 
                "x": (-0.5, 0.5),                           # 线速度减小，更缓慢的漂浮移动
                "y": (-0.5, 0.5),
                # "z": (0, 0),                           # 可以加一点上下漂浮
            },
            "overwrite_velocity" : False,                # True 不叠加
            "position_range": {
                "x": (-30.0, 30.0),
                "y": (-30.0, 30.0),
                # "z": (-1.0, 1.0),                            # 垂直方向上下浮动范围不宜太大
                # "yaw": (-0.2, 0.2),
                # "roll": (-0.2, 0.2),
                # "pitch": (-0.2, 0.2),
            },
        },
    )


    # # # # GMY
    # # 给平台添加一个事件，线速度角速度
    # push_platform = EventTerm(
    #     func=mdp.move_velocity,
    #     mode="interval",
    #     interval_range_s=(1, 1),                         # 每1.5~2.5秒触发一次，更加接近漂浮节奏
    #     params={
    #         "asset_cfg": SceneEntityCfg("platform"),
    #         "velocity_range": {
    #             "yaw": (0.5, 0.5),                        # 角速度减小，避免旋转太快
    #             "roll": (0.5, 0.5), 
    #             "pitch": (0.5, 0.5), 
    #             # "x": (-0.5, 0.5),                           # 线速度减小，更缓慢的漂浮移动
    #             # "y": (-0.5, 0.5),
    #             # "z": (0, 0),                           # 可以加一点上下漂浮
    #         },
    #         "overwrite_velocity" : True,                # True 不叠加
    #         "position_range": {
    #             # "x": (-20.0, 20.0),
    #             # "y": (-20.0, 20.0),
    #             # "z": (-1.0, 1.0),                            # 垂直方向上下浮动范围不宜太大
    #             # "yaw": (-0.2, 0.2),
    #             # "roll": (-0.2, 0.2),
    #             # "pitch": (-0.2, 0.2),
    #         },
    #     },
    # )


    # 给平台添加扰动加速度事件
    push_platform_acc = EventTerm(
        func=mdp.move_acceleration,
        mode="interval",
        interval_range_s=(0.02, 0.02),    # 每5ms施加一次扰动加速度（建议保持小周期，连续扰动）
        params={
            "asset_cfg": SceneEntityCfg("platform"),
        }
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # GMY 这个就是角速度的惩罚项（可以增大这个）
    # base_ang_acc_xy_l2 = RewTerm(func=mdp.base_ang_acc_xy_l2, weight=-1.0e-4)  # 调整权重控制惩罚强度
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)  # -0.05
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)  
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """粗糙地形上的速度跟踪任务"""

    # 场景配置
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # 观测、动作和指令
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # 马尔可夫决策过程配置
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""

        # 一般设置
        self.decimation = 4                                     # GMY 动作执行的降采样率，表示环境在每 4 步物理仿真后才更新一次动作（这个调大可能也能让角速度变平滑）
        self.episode_length_s = 100000                             # GMY(训练测试要改)单个 episode（训练回合）的最大时间： 20 秒
        # 仿真设置
        self.sim.dt = 0.005                                     # 物理仿真的时间步长： 0.005 秒
        self.sim.render_interval = self.decimation              # 渲染间隔： 与动作执行的降采样率相同
        self.sim.physics_material = self.scene.terrain.physics_material                 # 物理仿真使用地形的物理材质 
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15                           # 最大刚体补丁数量  
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
