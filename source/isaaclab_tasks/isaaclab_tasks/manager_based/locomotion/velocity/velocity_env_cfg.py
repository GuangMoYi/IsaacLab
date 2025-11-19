# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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


# GMY changed
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
            size=(100, 100, 10),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,                                                                  # 受重力影响(False)
                enable_gyroscopic_forces=True,                                                          # 允许自由旋转
                linear_damping=0.0,
                angular_damping=0.0,
                # kinematic_enabled = True,                                                             # 运动学物体，不受力
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=100000.0),                                      # 质量：降低到合理范围
            collision_props=sim_utils.CollisionPropertiesCfg(),               # 碰撞属性 
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
            # 恢复正常速度命令范围，让机器狗能够正常运动
            # 注意：机器狗需要先学会在平台上正常走路，然后再学习跟随平台
            lin_vel_x=(-1.5, 2.0),      # 前进速度范围（m/s）
            lin_vel_y=(-1.0, 1.0),      # 侧向速度范围（m/s）
            ang_vel_z=(-1.5, 1.5),      # 角速度范围（rad/s）
            heading=(-math.pi, math.pi)  # 朝向范围
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


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
        
        # ========== 平台观测（两种模式：延迟预测 vs 上帝视角） ==========
        # 模式1：延迟预测模式（使用神经网络预测，需要历史数据和预测）
        # 注意：当前使用"上帝视角"模式，下面的延迟预测观测已注释
        # 平台历史姿态（t-5之前的roll和pitch）
        # platform_history_orientation = ObsTerm(
        #     func=mdp.platform_history_orientation,
        #     params={
        #         "delay_steps": 5,  # 使用t-5之前的数据
        #         "history_length": 10,  # 历史长度
        #     },
        #     noise=Unoise(n_min=0, n_max=0),
        # )
        # # 平台历史角速度（t-5之前的roll和pitch角速度）
        # platform_history_angular_velocity = ObsTerm(
        #     func=mdp.platform_history_angular_velocity,
        #     params={
        #         "delay_steps": 5,  # 使用t-5之前的数据
        #         "history_length": 10,  # 历史长度
        #     },
        #     noise=Unoise(n_min=0, n_max=0),
        # )
        # # 预测的当前平台姿态（基于t-5之前的数据预测当前时刻）
        # platform_predicted_orientation = ObsTerm(
        #     func=mdp.platform_predicted_orientation,
        #     params={
        #         "delay_steps": 5,  # 使用t-5之前的数据预测当前时刻
        #     },
        #     noise=Unoise(n_min=0, n_max=0),
        # )
        # # 预测的当前平台角速度（基于t-5之前的数据预测当前时刻）
        # platform_predicted_angular_velocity = ObsTerm(
        #     func=mdp.platform_predicted_angular_velocity,
        #     params={
        #         "delay_steps": 5,  # 使用t-5之前的数据预测当前时刻
        #     },
        #     noise=Unoise(n_min=0, n_max=0),
        # )
        
        # 模式2：上帝视角模式（直接观测当前平台状态，无延迟，用于对比实验）
        # 关键改进：即使使用"上帝视角"，也需要添加历史观测，让机器狗学习运动规律
        # 当前平台姿态（上帝视角，直接使用当前时刻的平台roll和pitch）
        platform_current_orientation = ObsTerm(
            func=mdp.platform_current_orientation,
            noise=Unoise(n_min=0, n_max=0),
        )
        # 当前平台角速度（上帝视角，直接使用当前时刻的平台roll和pitch角速度）
        platform_current_angular_velocity = ObsTerm(
            func=mdp.platform_current_angular_velocity,
            noise=Unoise(n_min=0, n_max=0),
        )
        # 关键：添加平台历史观测，让机器狗能够学习运动规律
        # 平台历史姿态（最近N个时刻的roll和pitch，用于学习运动模式）
        platform_history_orientation = ObsTerm(
            func=mdp.platform_history_orientation,
            params={
                "delay_steps": 0,  # 使用当前时刻之前的数据（无延迟）
                "history_length": 20,  # 历史长度：20步（约0.4秒，足够学习正弦运动规律）
            },
            noise=Unoise(n_min=0, n_max=0),
        )
        # 平台历史角速度（最近N个时刻的roll和pitch角速度）
        platform_history_angular_velocity = ObsTerm(
            func=mdp.platform_history_angular_velocity,
            params={
                "delay_steps": 0,  # 使用当前时刻之前的数据（无延迟）
                "history_length": 20,  # 历史长度：20步
            },
            noise=Unoise(n_min=0, n_max=0),
        )
        # ============================================================================


        # 允许注入观测扰动（比如噪声、传感器掉线等鲁棒训练用手段）
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    @configclass
    class DebugCfg(ObsGroup):
        # platform_ang_acc = ObsTerm(func=mdp.platform_ang_acc_w)
        # platform_lin_acc = ObsTerm(func=mdp.platform_lin_acc_w)
        # platform_ang_vel = ObsTerm(func=mdp.platform_ang_vel_w)
        # platform_lin_vel = ObsTerm(func=mdp.platform_lin_vel_w)
        # platform_ang_w = ObsTerm(func=mdp.platform_ang_w)
        # platform_pos_w = ObsTerm(func=mdp.platform_pos_w)
        # robot_ang_acc_w = ObsTerm(func=mdp.robot_ang_acc_w)
        # robot_lin_acc_w = ObsTerm(func=mdp.robot_lin_acc_w)
        
        # ========== 新增：相对静止评估指标 ==========
        # 基座与平台姿态误差（rad）- 值越小表示基座与平台越平行
        # 这是机器狗基座XY平面和平台XY平面之间的误差
        base_platform_orientation_error = ObsTerm(func=mdp.base_platform_orientation_error_metric)
        
        # 平台自身姿态误差（rad）- 用于对比，应该接近0
        # 这是平台XY平面和水平0面的误差
        platform_orientation_error = ObsTerm(func=mdp.platform_orientation_error_metric)
        
        # 机器人相对于平台的角速度误差（rad/s）- 值越小表示角速度越同步
        robot_relative_ang_vel_error = ObsTerm(func=mdp.robot_relative_ang_vel_error_metric)
        
        # 机器狗误差和平台误差的比值（用于评估跟随效果）
        # 比值 = 机器狗基座与平台姿态误差 / 平台自身姿态误差
        # 比值越小，说明机器狗跟随效果越好（机器狗误差相对于平台误差很小）
        orientation_error_ratio = ObsTerm(func=mdp.orientation_error_ratio_metric)
        # ========== 新增：机器狗运动指标 ==========
        # 机器狗线速度大小（m/s）- 用于监控机器狗是否在运动
        robot_lin_vel_magnitude = ObsTerm(func=mdp.robot_lin_vel_magnitude)
        # 机器狗角速度大小（rad/s）- 用于监控机器狗是否在运动
        robot_ang_vel_magnitude = ObsTerm(func=mdp.robot_ang_vel_magnitude)
        # 机器狗线速度（世界坐标系，m/s）
        robot_lin_vel_w = ObsTerm(func=mdp.robot_lin_vel_w)
        # 机器狗角速度（世界坐标系，rad/s）
        robot_ang_vel_w = ObsTerm(func=mdp.robot_ang_vel_w)
        # ============================================

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

    if SceneEntityCfg("platform") is not None:
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


        # 给平台添加扰动加速度事件
        push_platform_acc = EventTerm(
            func=mdp.move_acceleration,
            mode="interval",
            interval_range_s=(0.02, 0.02),    # 每0.02s施加一次扰动加速度（建议保持小周期，连续扰动）
            params={
                "asset_cfg": SceneEntityCfg("platform"),
            }
        )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task (保持速度跟踪，让机器狗能够运动)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.75, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # 关键修改：禁用ang_vel_xy_l2惩罚，因为机器狗需要roll和pitch角速度来跟随平台运动
    # 如果惩罚roll/pitch角速度，机器狗无法调整姿态来跟随平台
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0.0)  # 改为0.0，不禁用但权重为0 原来是-0.05
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
    # -- optional penalties (权重为0，其他配置文件可以覆盖)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)
    
    # -- platform following reward (核心功能：让机器狗跟随平台运动)
    # 注意：权重需要与其他奖励平衡，不能太高导致忽略基本运动能力
    # 初期机器狗需要先学会基本平衡和运动，所以跟随奖励权重不能太高
    platform_following_with_history = RewTerm(
        func=mdp.platform_following_with_history_exp,
        weight=10.0,  # 提高权重：从5.0提高到10.0，让平台跟随成为主要任务
        params={
            "std_orientation": 0.08,  # 减小std：从0.2减小到0.08，使奖励函数更"尖锐"，在小误差时梯度更大
            # 当误差=0.065时，奖励=exp(-0.065/0.08)≈0.44（之前是0.72），梯度更大
            # 当误差=0.035时，奖励=exp(-0.035/0.08)≈0.65（之前是0.84），仍有足够奖励
            "std_angular_velocity": 0.1,  # 角速度误差的标准差（平台最大角速度约0.016 rad/s，设置为0.1可以覆盖合理范围）
            "prediction_horizon": 0.2,
            "history_length": 50,
            "use_god_view": True,  # 是否使用"上帝视角"（直接使用当前平台状态，无延迟）
            # 设置为True：机器狗可以直接观测到当前平台状态，用于对比实验
            # 设置为False：使用神经网络预测平台状态（需要延迟观测和预测）
        }
    )


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
    
    # 注意：不再使用课程学习来调整跟随平台奖励权重
    # 现在使用延迟观测和预测的方法，机器狗可以直接学习跟随平台运动


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
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
