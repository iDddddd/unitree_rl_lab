import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp

# ----------------------------------------------------------------------------
# 平台运动环境配置（Unitree-G1-29DOF-PlatformVelocity）
#
# 这个文件定义了一个新的机器人行走任务：
# - 机器人位于可移动的平台（kinematic platform）上
# - 平台会按照正弦曲线在水平面内运动（包含平移与旋转分量）
# - 机器人需要适应平台运动，执行基于速度跟踪的 locomotion 目标
# - 终止条件包括倒地、高倾斜、超时、以及“是否离开平台”（可配置为软、硬）
#
# 核心模块说明：
# - RobotSceneCfg：定义场景物理对象（地面、平台、机器人、传感器）
# - EventCfg：定义训练过程中触发的事件（随机化、重置、平台运动）
# - CommandsCfg / ActionsCfg / ObservationsCfg / RewardsCfg / TerminationsCfg：
#   MDP 组件，分别定义命令、动作空间、观测项、奖励函数、终止规则
# - CurriculumCfg：定义课程学习策略（逐步增加速度命令、平台动作难度等）
# - RobotEnvCfg：汇总所有配置，设置仿真参数、环境规模、节点周期等
#-----------------------------------------------------------------------------

PLATFORM_SIZE_X = 20.0
PLATFORM_SIZE_Y = 20.0 # 平台尺寸，确保足够大以容纳机器人在上面运动，同时也可以调整以增加或减少运动难度
PLATFORM_THICKNESS = 0.2 # 平台厚度，设置为0.2米以确保平台在物理模拟中具有足够的厚度，避免穿透问题，同时也不会过高以影响机器人运动的真实性
PLATFORM_TOP_Z = 1.0 # 平台顶部的高度，设置为1.0米以提供足够的空间让机器人在平台上运动，同时也可以调整以增加或减少运动难度

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(100.0, 100.0), # 生成的平台尺寸，设置为10x10米以提供足够的空间让机器人在上面运动
    border_width=20.0, # 平台边界宽度，设置为20米以确保机器人在接近边界时能够感受到边界的存在，同时也可以调整以增加或减少运动难度
    num_rows=9, # 生成的石块行数，设置为9行以提供适度的复杂性，同时也可以调整以增加或减少运动难度
    num_cols=21, # 生成的石块列数，设置为21列以提供适度的复杂性，同时也可以调整以增加或减少运动难度
    horizontal_scale=0.1, # 水平高度变化的缩放比例，设置为0.1以提供适度的高度变化，同时也可以调整以增加或减少运动难度
    vertical_scale=0.005, # 垂直高度变化的缩放比例，设置为0.005以提供适度的高度变化，同时也可以调整以增加或减少运动难度
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",  # "plane", "generator" ，这里选择 "plane" 以保持地面平坦，突出平台运动的挑战；如果选择 "generator"，则会在平台上生成不平坦的地形，增加额外的难度。
        terrain_generator=None, # COBBLESTONE_ROAD_CFG, # 如果 terrain_type 是 "generator"，则使用这个配置生成地形；如果 terrain_type 是 "plane"，则忽略这个配置。
        collision_group=-1, # 设置为 -1 以确保地面不会与机器人发生碰撞，避免干扰平台运动的挑战；如果需要地面与机器人发生碰撞，可以设置为其他值。
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", # 摩擦力结合模式，设置为 "multiply" 以提供更真实的摩擦行为，增强训练的稳定性和效果；如果需要更简单的摩擦模型，可以选择 "average" 或 "min"。
            restitution_combine_mode="multiply", # 恢复力结合模式，设置为 "multiply" 以提供更真实的碰撞恢复行为，增强训练的稳定性和效果；如果需要更简单的恢复模型，可以选择 "average" 或 "min"。
            static_friction=1.0, # 静摩擦系数，设置为 1.0 以提供足够的摩擦力，帮助机器人在平台上保持稳定；如果需要更滑的表面，可以降低这个值。
            dynamic_friction=1.0, # 动摩擦系数，设置为 1.0 以提供足够的摩擦力，帮助机器人在平台上保持稳定；如果需要更滑的表面，可以降低这个值。
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25), # 纹理缩放，设置为 0.25 以提供更细腻的纹理细节，增强视觉效果；如果需要更大块的纹理，可以增加这个值。
        ),
        debug_vis=False, # 是否启用调试可视化，设置为 False 以减少视觉干扰，帮助训练更专注于平台运动的挑战；如果需要调试地形生成，可以设置为 True。   
    )
    # kinematic platform (one per environment)
    platform: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Platform", # 平台的 Prim 路径，使用环境变量模板以支持多环境实例化
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, PLATFORM_TOP_Z - 0.5 * PLATFORM_THICKNESS),
        ), # 平台初始位置，设置在地面上方 PLATFORM_TOP_Z 米处，减去平台厚度的一半以确保平台顶部在 PLATFORM_TOP_Z 位置
        spawn=sim_utils.CuboidCfg(
            size=(PLATFORM_SIZE_X, PLATFORM_SIZE_Y, PLATFORM_THICKNESS), # 平台的尺寸，使用之前定义的常量
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ), # 设置平台为运动学物体，禁用重力，以确保平台按照预定的运动轨迹移动，而不受物理力的影响
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.4)), # 平台的视觉材质，设置为预览表面并指定颜色，以提供清晰的视觉区分，帮助训练更专注于平台运动的挑战；如果需要更复杂的视觉效果，可以使用 MdlFileCfg。
        ),
    )

    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot", # 机器人的 Prim 路径，使用环境变量模板以支持多环境实例化
        init_state=ROBOT_CFG.init_state.replace(
            pos=(
                ROBOT_CFG.init_state.pos[0],
                ROBOT_CFG.init_state.pos[1],
                ROBOT_CFG.init_state.pos[2] + PLATFORM_TOP_Z,
            )
        ), # 机器人初始位置，设置在平台顶部，确保机器人在平台上开始训练
    )

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link", # 高度扫描器的 Prim 路径，安装在机器人躯干链接上，以测量机器人与平台之间的距离，帮助训练更好地适应平台运动的挑战
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)), # 扫描器偏移位置，设置在躯干链接上方 20 米处，以确保能够扫描到平台，即使在机器人跳跃或平台运动较大时也能保持测量稳定
        ray_alignment="yaw", # 射线对齐方式，设置为 "yaw" 以确保扫描器始终朝向平台的法线方向，提供更稳定的高度测量；如果需要固定方向，可以选择 "none"。
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]), # 扫描模式配置，使用网格模式以提供更全面的高度信息，分辨率设置为 0.1 米以提供足够的细节，扫描区域设置为 1.6x1.0 米以覆盖机器人底部区域；如果需要更大或更小的扫描范围，可以调整 size 参数。
        debug_vis=False,
        # RayCaster requires a global absolute prim path (starting with '/').
        # Per-env template paths like "{ENV_REGEX_NS}/..." are not supported here.
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True) # 接触力传感器配置，安装在机器人所有链接上，以测量机器人与平台之间的接触力，帮助训练更好地适应平台运动的挑战；history_length 设置为 3 以提供短期的接触历史，track_air_time 设置为 True 以跟踪机器人离地时间，提供更多关于机器人状态的信息。
    left_foot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        history_length=3,
        track_air_time=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Platform"],
    )
    right_foot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        history_length=3,
        track_air_time=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Platform"],
    )
    base_imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=ImuCfg.OffsetCfg(pos=(-0.03959, -0.00224, 0.14792)),
        gravity_bias=(0.0, 0.0, 0.0),
        debug_vis=False,
    )
    left_foot_imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        offset=ImuCfg.OffsetCfg(pos=(0.035, 0.0, -0.03)),
        gravity_bias=(0.0, 0.0, 0.0),
        debug_vis=False,
    )
    right_foot_imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        offset=ImuCfg.OffsetCfg(pos=(0.035, 0.0, -0.03)),
        gravity_bias=(0.0, 0.0, 0.0),
        debug_vis=False,
    )
    # lights，设置一个环境范围内的全局光源，以提供均匀的照明，帮助训练更好地适应平台运动的挑战；如果需要更复杂的照明效果，可以添加更多的光源或使用不同类型的光源。
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup，在环境启动时触发的事件，用于随机化物理属性等。
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"), # 应用于机器人的所有链接，以增加训练的多样性和鲁棒性；如果需要更有针对性的随机化，可以指定特定的链接。
            "static_friction_range": (0.3, 1.0), # 静摩擦系数的随机范围，设置为 (0.3, 1.0) 以提供适度的摩擦力，帮助训练更好地适应平台运动的挑战；如果需要更滑或更粘的表面，可以调整这个范围。
            "dynamic_friction_range": (0.3, 1.0), # 动摩擦系数的随机范围，设置为 (0.3, 1.0) 以提供适度的摩擦力，帮助训练更好地适应平台运动的挑战；如果需要更滑或更粘的表面，可以调整这个范围。
            "restitution_range": (0.0, 0.0), # 恢复力的随机范围，设置为 (0.0, 0.0) 以禁用恢复力，避免机器人在平台上弹跳过高，增加训练的稳定性；如果需要更弹性的表面，可以增加这个范围。
            "num_buckets": 64,  # 随机化参数的离散化桶数量，设置为 64 以提供足够的随机化粒度，帮助训练更好地适应平台运动的挑战；如果需要更细或更粗的随机化，可以调整这个值。
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass, # 在环境启动时随机增加机器人基座的质量，以增加训练的多样性和鲁棒性；如果需要更有针对性的随机化，可以指定其他链接或调整质量范围。
        mode="startup", 
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },# 这里设置 mass_distribution_params 的范围为 (-1.0, 3.0)，允许在原有质量基础上增加最多 3.0 kg 的质量，同时也允许减少最多 1.0 kg 的质量，以提供更广泛的随机化效果；如果需要更保守或更激进的随机化，可以调整这个范围。
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque, # 在环境重置时对机器人基座施加随机的外部力和力矩，以增加训练的多样性和鲁棒性；如果需要更有针对性的随机化，可以指定其他链接或调整力/力矩范围。
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        }, # 这里设置 force_range 和 torque_range 都为 (0.0, 0.0)，表示在重置时不施加额外的外部力和力矩，以提供一个相对稳定的起始状态；如果需要更具挑战性的重置，可以增加这些范围。
    )

    reset_platform = EventTerm(
        func=mdp.reset_platform_state, # 在机器人重置前，先将平台恢复到默认位姿并清零速度，同时重置该环境的平台运动相位，避免机器人生成时与抬升/倾斜的平台发生穿插碰撞。
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("platform"),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform, # 在环境重置时将机器人基座的状态随机重置在一个范围内，以增加训练的多样性和鲁棒性；如果需要更有针对性的重置，可以调整位置和姿态的范围。
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }, # 这里设置 pose_range 的位置范围为 (-0.5, 0.5) 米，yaw 范围为 (-3.14, 3.14) 弧度，允许机器人在平台上有一定的随机位置和朝向；velocity_range 设置为 (0.0, 0.0) 表示在重置时不赋予额外的初始速度，以提供一个相对稳定的起始状态；如果需要更具挑战性的重置，可以增加这些范围。
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale, # 在环境重置时将机器人关节位置随机重置在一个范围内，以增加训练的多样性和鲁棒性；如果需要更有针对性的重置，可以调整关节位置的缩放范围。
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        }, # 这里设置 position_range 为 (1.0, 1.0)，表示在重置时将关节位置随机重置在默认位置的基础上增加最多 100% 的偏移，以提供更广泛的随机化效果；velocity_range 设置为 (-1.0, 1.0) 表示在重置时将关节速度随机重置在一个范围内，以提供更多的初始状态多样性；如果需要更保守或更激进的随机化，可以调整这些范围。
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity, # 在环境运行期间定期对机器人施加随机的推力，以增加训练的多样性和鲁棒性；如果需要更有针对性的随机化，可以调整力的范围。
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )# 这里设置 interval_range_s 为 (5.0, 5.0)，表示每隔 5 秒对机器人施加一次推力，以提供定期的挑战；velocity_range 设置为 (-0.5, 0.5) m/s，表示推力将使机器人在 x 和 y 方向上获得一个随机的速度增量，以增加训练的多样性和鲁棒性；如果需要更频繁或更稀疏的推力，可以调整 interval_range_s；如果需要更强或更弱的推力，可以调整 velocity_range。

    move_platform = EventTerm(
        func=mdp.move_platform_sine,# 在环境运行期间定期按照正弦曲线移动平台，以增加训练的挑战性和多样性；如果需要更有针对性的运动模式，可以调整运动函数或参数。
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "asset_cfg": SceneEntityCfg("platform"),
            "lin_frequency_hz": 0.2,
            "max_linear_acc": 0.5,
            # at 4m radius (half platform size), 0.125 rad/s^2 -> 0.5 m/s^2 tangential acceleration
            "max_angular_acc": 0.05,
        },# 这里设置 interval_range_s 为 (0.02, 0.02)，表示每隔 0.02 秒更新一次平台的位置，以提供连续的运动挑战；lin_frequency_hz 设置为 0.2 Hz，表示平台将以 0.2 Hz 的频率进行正弦运动；max_linear_acc 设置为 0.5 m/s^2，表示平台的线性加速度将被限制在这个值，以确保运动的平滑性和可控性；max_angular_acc 设置为 0.125 rad/s^2，表示平台的角加速度将被限制在这个值，以确保旋转运动的平滑性和可控性；如果需要更快或更慢的运动频率，可以调整 lin_frequency_hz；如果需要更强或更弱的运动幅度，可以调整 max_linear_acc 和 max_angular_acc。
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # 机器人基座速度命令，采样方式为 “UniformLevel” 即按 level 选择区间。
    # asset_name：命令关联的机器人资产。
    # resampling_time_range：重新采样新命令的时间区间（秒），这里固定10秒一次。
    # rel_standing_envs：在站立环境中的比例，可能影响选取静止命令概率。
    # rel_heading_envs：与朝向相关的环境比例。
    # heading_command：是否包含航向命令，此处关闭 (False)。
    # debug_vis：是否可视化命令，将在训练时显示目标方向。
    # ranges：当前命令区间，初始都为0（无运动），配合 curriculum 动态升级。
    # limit_ranges：可允许指令最大范围，最终可达 +/-1m/s， +/-0.2rad/s。
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.2, 0.2)
        ),
    ) # 这里设置 base_velocity 命令的 resampling_time_range 为 (10.0, 10.0)，表示每隔 10 秒重新采样一次新的速度命令；rel_standing_envs 设置为 0.02，表示在站立环境中有 2% 的概率选择静止命令；rel_heading_envs 设置为 1.0，表示所有环境都包含朝向相关的命令；heading_command 设置为 False，表示不包含独立的航向命令；debug_vis 设置为 True，表示在训练时可视化显示目标方向；ranges 设置了初始的命令区间，初始都为 (-0.1, 0.1) m/s 或 rad/s，表示初始命令较小以适应训练初期；limit_ranges 设置了可允许的最大命令范围，最终可达 (-0.5, 1.0) m/s 的线速度和 (-0.2, 0.2) rad/s 的角速度，以提供足够的运动挑战；如果需要更频繁或更稀疏的命令更新，可以调整 resampling_time_range；如果需要更强或更弱的运动挑战，可以调整 ranges 和 limit_ranges。


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # JointPositionAction: 以关节位置增量作为 policy 输出。
    # asset_name：作用对象。
    # joint_names：使用正则选择所有关节。
    # scale：动作尺度，0.25 档位，控制输出幅度。
    # use_default_offset：是否使用默认偏置值（True）。
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )# 这里设置 JointPositionAction 的 scale 为 0.25，表示 policy 输出的关节位置增量将被缩放为原来的 25%，以提供适度的动作幅度，帮助训练更好地适应平台运动的挑战；use_default_offset 设置为 True，表示使用默认的偏置值，这通常是机器人当前的关节位置，以提供一个合理的动作基准；如果需要更大或更小的动作幅度，可以调整 scale；如果需要使用不同的偏置策略，可以调整 use_default_offset。


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    # 这里定义了 policy 和 critic 两组观测，分别用于 policy 网络和 critic 网络的输入。
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-20.0, 20.0), noise=Unoise(n_min=-0.2, n_max=0.2)) # 机器人基座角速度观测，添加噪声以增加训练的鲁棒性；scale 设置为 0.2 以缩放观测值，noise 设置为 Uniform(-0.2, 0.2) 以提供适度的观测噪声，帮助训练更好地适应平台运动的挑战；如果需要更精确或更嘈杂的观测，可以调整 scale 和 noise。
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-5.0, 5.0), noise=Unoise(n_min=-0.05, n_max=0.05)) # 机器人重力投影观测，添加噪声以增加训练的鲁棒性；noise 设置为 Uniform(-0.05, 0.05) 以提供适度的观测噪声，帮助训练更好地适应平台运动的挑战；如果需要更精确或更嘈杂的观测，可以调整 noise。
        ekf_base_pos_rel_platform = ObsTerm(
            func=mdp.ekf_base_pos_rel_platform,
            clip=(-5.0, 5.0),
            noise=Unoise(n_min=-0.02, n_max=0.02),
        ) # EKF 估计的机身相对平台位置观测。
        ekf_base_vel_rel_platform = ObsTerm(
            func=mdp.ekf_base_vel_rel_platform,
            clip=(-10.0, 10.0),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ) # EKF 估计的机身相对平台速度观测。
        ekf_base_quat_rel_platform = ObsTerm(
            func=mdp.ekf_base_quat_rel_platform,
            clip=(-1.0, 1.0),
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ) # EKF 估计的机身相对平台姿态观测。
        velocity_commands = ObsTerm(func=mdp.generated_commands, clip=(-5.0, 5.0), params={"command_name": "base_velocity"}) # 机器人当前速度命令观测，直接使用生成的命令作为观测项，以提供清晰的目标信息，帮助训练更好地适应平台运动的挑战；如果需要更复杂的命令表示，可以添加额外的处理或特征提取。
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-10.0, 10.0), noise=Unoise(n_min=-0.01, n_max=0.01)) # 机器人关节位置相对观测，添加噪声以增加训练的鲁棒性；noise 设置为 Uniform(-0.01, 0.01) 以提供适度的观测噪声，帮助训练更好地适应平台运动的挑战；如果需要更精确或更嘈杂的观测，可以调整 noise。
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-20.0, 20.0), noise=Unoise(n_min=-1.5, n_max=1.5)) # 机器人关节速度相对观测，添加噪声以增加训练的鲁棒性；scale 设置为 0.05 以缩放观测值，noise 设置为 Uniform(-1.5, 1.5) 以提供较大的观测噪声，帮助训练更好地适应平台运动的挑战；如果需要更精确或更嘈杂的观测，可以调整 scale 和 noise。
        last_action = ObsTerm(func=mdp.last_action, clip=(-10.0, 10.0)) # 机器人最后执行的动作观测，用于提供动作历史信息，帮助训练更好地适应平台运动的挑战；如果需要更复杂的动作历史表示，可以添加额外的处理或特征提取。
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8}) # 机器人步态相位观测，基于一个周期为 0.8 秒的正弦函数计算步态相位，以提供关于机器人运动周期的信息，帮助训练更好地适应平台运动的挑战；如果需要更复杂的步态表示，可以添加额外的处理或特征提取。

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True # 是否将所有观测项连接成一个大向量，设置为 True 以提供一个统一的观测表示，帮助训练更好地适应平台运动的挑战；如果需要分组或分开处理观测项，可以设置为 False。

    # observation groups
    policy: PolicyCfg = PolicyCfg() # 这里定义了 policy 观测组，包含了机器人基座角速度、重力投影、当前速度命令、关节位置相对观测、关节速度相对观测、最后执行的动作等观测项，这些观测项提供了关于机器人状态和目标的信息，帮助训练更好地适应平台运动的挑战；如果需要更多或更少的观测项，可以调整 PolicyCfg 中的定义。

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-20.0, 20.0)) # 机器人基座线速度观测
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-20.0, 20.0)) # 机器人基座角速度观测
        platform_body_vel_deltas = ObsTerm(
            func=mdp.platform_body_vel_deltas_b,
            clip=(-20.0, 20.0),
            params={
                "robot_asset_cfg": SceneEntityCfg("robot"),
                "platform_asset_cfg": SceneEntityCfg("platform"),
            },
        ) # critic 额外特权观测：平台-机体相对速度 [v_xy^B, w_z^B]，用于稳定价值估计。
        ekf_base_pos_rel_platform = ObsTerm(
            func=mdp.gt_base_pos_rel_platform,
            clip=(-5.0, 5.0),
            params={
                "robot_asset_cfg": SceneEntityCfg("robot"),
                "platform_asset_cfg": SceneEntityCfg("platform"),
            },
        ) # 特权真值：机身相对平台位置（平台坐标系）。
        ekf_base_vel_rel_platform = ObsTerm(
            func=mdp.gt_base_vel_rel_platform,
            clip=(-10.0, 10.0),
            params={
                "robot_asset_cfg": SceneEntityCfg("robot"),
                "platform_asset_cfg": SceneEntityCfg("platform"),
            },
        ) # 特权真值：机身相对平台速度（平台坐标系）。
        ekf_base_quat_rel_platform = ObsTerm(
            func=mdp.gt_base_quat_rel_platform,
            clip=(-1.0, 1.0),
            params={
                "robot_asset_cfg": SceneEntityCfg("robot"),
                "platform_asset_cfg": SceneEntityCfg("platform"),
            },
        ) # 特权真值：机身相对平台姿态四元数。
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-5.0, 5.0)) # 机器人重力投影观测
        velocity_commands = ObsTerm(func=mdp.generated_commands, clip=(-5.0, 5.0), params={"command_name": "base_velocity"}) # 机器人当前速度命令观测
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-10.0, 10.0)) # 机器人关节位置相对观测
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-20.0, 20.0)) # 机器人关节速度相对观测
        last_action = ObsTerm(func=mdp.last_action, clip=(-10.0, 10.0)) # 机器人最后执行的动作观测
        # gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8})
        # height_scanner = ObsTerm(func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     clip=(-1.0, 5.0),
        # )

        def __post_init__(self):
            self.history_length = 5 # 这里设置了 critic 观测组的 history_length 为 5，表示 critic 将使用最近 5 个时间步的观测历史来进行价值估计，以提供更多的时间上下文信息，帮助训练更好地适应平台运动的挑战；如果需要更短或更长的历史，可以调整这个值。

    # privileged observations
    critic: CriticCfg = CriticCfg()

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # 这里定义了多个奖励项，涵盖了任务目标、机器人状态、足部接触等方面，以提供一个综合的奖励信号，帮助训练更好地适应平台运动的挑战；如果需要更简单或更复杂的奖励结构，可以调整这些奖励项。
    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )# 这里设置 track_lin_vel_xy 的 func 为 mdp.track_lin_vel_xy_yaw_frame_exp，表示使用基于机器人朝向的线速度跟踪奖励函数，以提供更准确的速度跟踪信号，帮助训练更好地适应平台运动的挑战；weight 设置为 1.0，表示这个奖励项在总奖励中的权重较高，以强调任务目标的重要性；params 中的 command_name 设置为 "base_velocity"，表示这个奖励项将跟踪 base_velocity 命令；std 设置为 sqrt(0.25)，表示奖励函数中的误差将被缩放为原来的 0.5，以提供适度的奖励信号，帮助训练更好地适应平台运动的挑战；如果需要更强或更弱的奖励信号，可以调整 weight 和 std。
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    ) # 这里设置 track_ang_vel_z 的 func 为 mdp.track_ang_vel_z_exp，表示使用基于机器人朝向的角速度跟踪奖励函数，以提供更准确的角速度跟踪信号，帮助训练更好地适应平台运动的挑战；weight 设置为 0.5，表示这个奖励项在总奖励中的权重较高，但不如线速度跟踪重要，以强调任务目标的重要性；params 中的 command_name 设置为 "base_velocity"，表示这个奖励项将跟踪 base_velocity 命令；std 设置为 sqrt(0.25)，表示奖励函数中的误差将被缩放为原来的 0.5，以提供适度的奖励信号，帮助训练更好地适应平台运动的挑战；如果需要更强或更弱的奖励信号，可以调整 weight 和 std。

    alive = RewTerm(func=mdp.is_alive, weight=0.2)# 这里设置 alive 的 func 为 mdp.is_alive，表示使用一个简单的存活奖励函数，提供一个二元奖励信号，帮助训练保持机器人在平台上；weight 设置为 0.2，表示这个奖励项在总奖励中的权重较低，但仍然重要，以鼓励机器人保持存活状态；如果需要更强或更弱的存活激励，可以调整 weight。

    # -- base
    base_linear_velocity = RewTerm(
        func=mdp.base_lin_vel_z_rel_platform_l2, weight=-0.2
    ) # 机器人基座线速度奖励（相对平台 z 速度），避免将平台垂向运动误判为机器人错误。
    base_angular_velocity = RewTerm(
        func=mdp.base_ang_vel_xy_rel_platform_l2, weight=-0.02
    ) # 机器人基座角速度奖励（相对平台 roll/pitch 角速度），减少平台旋转带来的误罚。
    relative_platform_velocity = RewTerm(
        func=mdp.platform_body_vel_deltas_l2,
        weight=-0.05,
        params={
            "lin_xy_weight": 1.0,
            "ang_z_weight": 0.5,
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "platform_asset_cfg": SceneEntityCfg("platform"),
        },
    ) # 新增：惩罚平台与机体相对速度误差，强化对外界扰动的鲁棒性。
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001) # 机器人关节速度奖励，使用基于关节速度的 L2 奖励函数，以提供一个关于机器人动作平滑性的奖励信号，帮助训练更好地适应平台运动的挑战；weight 设置为 -0.001，表示这个奖励项在总奖励中的权重较低，并且是一个惩罚项，以鼓励机器人保持较低的关节速度；如果需要更强或更弱的惩罚信号，可以调整 weight。
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7) # 机器人关节加速度奖励，使用基于关节加速度的 L2 奖励函数，以提供一个关于机器人动作平滑性的奖励信号，帮助训练更好地适应平台运动的挑战；weight 设置为 -2.5e-7，表示这个奖励项在总奖励中的权重非常低，并且是一个惩罚项，以鼓励机器人保持较低的关节加速度；如果需要更强或更弱的惩罚信号，可以调整 weight。
    action_rate = RewTerm(
        func=mdp.action_rate_l2_bounded,
        weight=-0.03,
        params={"action_clip": 10.0, "reward_clip": 1.0e3},
    ) # 有界的动作变化率惩罚，避免单个坏 step 产生极端尖峰并污染 PPO 更新。
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0) # 机器人关节位置限制奖励，使用基于关节位置限制的奖励函数，以提供一个关于机器人动作可行性的奖励信号，帮助训练更好地适应平台运动的挑战；weight 设置为 -5.0，表示这个奖励项在总奖励中的权重较高，并且是一个惩罚项，以鼓励机器人保持在关节位置限制范围内；如果需要更强或更弱的惩罚信号，可以调整 weight。
    energy = RewTerm(func=mdp.energy, weight=-2e-5) # 机器人能量奖励，使用基于能量的奖励函数，以提供一个关于机器人效率的奖励信号，帮助训练更好地适应平台运动的挑战；weight 设置为 -2e-5，表示这个奖励项在总奖励中的权重较低，并且是一个惩罚项，以鼓励机器人保持较低的能量消耗；如果需要更强或更弱的惩罚信号，可以调整 weight。

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_joint",
                    ".*_wrist_.*",
                ],
            )
        },
    ) # 机器人手臂关节偏离奖励，使用基于关节位置偏离的 L1 奖励函数，以提供一个关于机器人动作自然性的奖励信号，帮助训练更好地适应平台运动的挑战；weight 设置为 -0.1，表示这个奖励项在总奖励中的权重较低，并且是一个惩罚项，以鼓励机器人保持手臂关节位置接近默认位置；params 中的 asset_cfg 使用正则表达式选择了肩部、肘部和腕部的关节，以专注于手臂部分；如果需要更强或更弱的惩罚信号，可以调整 weight；如果需要调整关注的关节，可以修改 joint_names 中的正则表达式。
    joint_deviation_waists = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist.*",
                ],
            )
        },
    ) # 机器人腰部关节偏离奖励，使用基于关节位置偏离的 L1 奖励函数，以提供一个关于机器人动作自然性的奖励信号，帮助训练更好地适应平台运动的挑战；weight 设置为 -1，表示这个奖励项在总奖励中的权重较高，并且是一个惩罚项，以鼓励机器人保持腰部关节位置接近默认位置；params 中的 asset_cfg 使用正则表达式选择了所有包含 "waist" 的关节，以专注于腰部部分；如果需要更强或更弱的惩罚信号，可以调整 weight；如果需要调整关注的关节，可以修改 joint_names 中的正则表达式。
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint", ".*_hip_yaw_joint"])},
    )# 机器人腿部关节偏离奖励，使用基于关节位置偏离的 L1 奖励函数，以提供一个关于机器人动作自然性的奖励信号，帮助训练更好地适应平台运动的挑战；weight 设置为 -1.0，表示这个奖励项在总奖励中的权重较高，并且是一个惩罚项，以鼓励机器人保持腿部关节位置接近默认位置；params 中的 asset_cfg 使用正则表达式选择了所有包含 "_hip_roll_joint" 和 "_hip_yaw_joint" 的关节，以专注于腿部部分；如果需要更强或更弱的惩罚信号，可以调整 weight；如果需要调整关注的关节，可以修改 joint_names 中的正则表达式。

    # -- robot
    flat_orientation_l2 = RewTerm(
        func=mdp.base_orientation_rel_platform_l2,
        weight=-1.0,
        params={
            "deadband_rad": 0.08,
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "platform_asset_cfg": SceneEntityCfg("platform"),
        },
    ) # 优化：改为相对平台姿态惩罚（roll/pitch），降低平台主动倾斜时的误罚。
    base_height = RewTerm(
        func=mdp.base_height_relative_to_platform_normal_l2,
        weight=-2.0,
        params={
            "target_height": 0.78,
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "platform_asset_cfg": SceneEntityCfg("platform"),
        },
    ) # 机器人基座高度奖励（相对平台），用相对高度替代世界系固定高度以避免误罚。

    # -- feet
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.8,
        params={
            "period": 0.7,
            "offset": [0.0, 0.5],
            "threshold": 0.5,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    ) # 机器人足部步态奖励，使用基于足部接触模式的奖励函数，以提供一个关于机器人步态协调性的奖励信号，帮助训练更好地适应平台运动的挑战；weight 设置为 0.5，表示这个奖励项在总奖励中的权重较高，以强调步态协调的重要性；params 中的 period 设置为 0.8 秒，表示步态周期；offset 设置为 [0.0, 0.5]，表示两个足部的相位偏移；threshold 设置为 0.55，表示接触力的阈值；command_name 设置为 "base_velocity"，表示这个奖励项将根据 base_velocity 命令进行计算；sensor_cfg 使用正则表达式选择了所有包含 "ankle_roll" 的接触力传感器，以专注于足部接触信息；如果需要更强或更弱的奖励信号，可以调整 weight；如果需要调整步态参数，可以修改 params 中的值。
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    ) # 机器人足部滑动奖励，使用基于足部滑动的奖励函数，以提供一个关于机器人足部稳定性的奖励信号，帮助训练更好地适应平台运动的挑战；weight 设置为 -0.2，表示这个奖励项在总奖励中的权重较低，并且是一个惩罚项，以鼓励机器人减少足部滑动；params 中的 asset_cfg 使用正则表达式选择了所有包含 "ankle_roll" 的身体部件，以专注于足部信息；sensor_cfg 使用正则表达式选择了所有包含 "ankle_roll" 的接触力传感器，以专注于足部接触信息；如果需要更强或更弱的惩罚信号，可以调整 weight；如果需要调整关注的身体部件或传感器，可以修改相应的正则表达式。
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_relative_platform_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "platform_asset_cfg": SceneEntityCfg("platform"),
        },
    ) # 机器人足部离地奖励，使用基于足部离地高度的奖励函数，以提供一个关于机器人步态质量的奖励信号，帮助训练更好地适应平台运动的挑战；weight 设置为 1.0，表示这个奖励项在总奖励中的权重较高，以强调足部离地的重要性；params 中的 std 设置为 0.05，表示奖励函数中的误差将被缩放为原来的 0.05 米，以提供适度的奖励信号；tanh_mult 设置为 2.0，表示使用双曲正切函数来计算奖励时的乘数，以调整奖励曲线的形状；target_height 设置为 0.1 米，表示奖励函数将以这个高度作为目标进行计算；asset_cfg 使用正则表达式选择了所有包含 "ankle_roll" 的身体部件，以专注于足部信息；如果需要更强或更弱的奖励信号，可以调整 weight；如果需要调整奖励函数参数，可以修改 params 中的值。
    feet_air_time_balance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")},
    ) # 抑制两脚腾空/落地时长差异过大，减少“一脚迈出后另一脚急追”现象。

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    ) # 机器人 undesired_contacts 奖励，使用基于 undesired_contacts 的奖励函数，以提供一个关于机器人与环境交互的奖励信号，帮助训练更好地适应平台运动的挑战；weight 设置为 -1，表示这个奖励项在总奖励中的权重较低，并且是一个惩罚项，以鼓励机器人减少与环境的不期望接触；params 中的 threshold 设置为 1，表示接触力的阈值；sensor_cfg 使用正则表达式选择了所有不包含 "ankle" 的身体部件，以专注于其他接触信息；如果需要更强或更弱的惩罚信号，可以调整 weight；如果需要调整关注的身体部件，可以修改相应的正则表达式。


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # 终止1：达到全局最大时间则结束。
    # 1) func：调用 mdp.time_out 进行计时判断。
    # 2) time_out=True：直接将其视为时间超时终止条件。
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 终止2：机器人身体高度过低（跌倒或贴地）时结束。
    # params.minimum_height: 机器人根关节高度阈值，低于就认为倒地。
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.2})

    # 终止3：机器人倾斜角过大时结束。
    # params.limit_angle: 机器人根关节与垂直方向夹角阈值（rad）。
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})

    # 终止4：机器人脱离平台边界时结束。
    # 1) func：mdp.outside_platform_bounds 计算与平台中心的平移距离是否超出半边长+margin。
    # 2) margin：允许的额外宽松范围，0.2米。
    # 3) half_size_x / half_size_y：平台半宽高，当前为平台大小一半。
    # 4) robot_asset_cfg / platform_asset_cfg：指定参与判断的机器人与平台实体。
    off_platform = DoneTerm(
        func=mdp.outside_platform_bounds,
        params={
            "margin": 0.2,
            "half_size_x": 1.0 * PLATFORM_SIZE_X,
            "half_size_y": 1.0 * PLATFORM_SIZE_Y, 
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "platform_asset_cfg": SceneEntityCfg("platform"),
        },
    ) # 这里设置 off_platform 的 func 为 mdp.outside_platform_bounds，表示使用一个基于机器人与平台边界关系的终止条件，以提供一个关于机器人是否脱离平台的判断，帮助训练更好地适应平台运动的挑战；params 中的 margin 设置为 0.2 米，表示在平台边界基础上允许的额外宽松范围；half_size_x 和 half_size_y 设置为平台大小；robot_asset_cfg 和 platform_asset_cfg 分别指定了参与判断的机器人和平台实体；如果需要更严格或更宽松的边界条件，可以调整 margin；如果需要适应不同大小的平台，可以调整 half_size_x 和 half_size_y。


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # 地形难度课程暂不启动（plane地形）。
    terrain_levels = None

    # 速度指令课程：这个函数由 mdp.lin_vel_cmd_levels 计算当前速度区间和级别。
    # 在训练早期保持较低命令速度，随着课程进度放开到 limit_ranges。
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)

    # 当前 episode 的平均进度（0~1），用于观察训练早期是否普遍提前终止。
    episode_count = CurrTerm(func=mdp.episode_count)

    # 平台运动级数课程：逐级增加平台运动自由度/幅度
    # params:
    #   - motion_mode：4种训练模式之一
    #       "rpy" 仅旋转 DoF，按 roll -> pitch -> yaw 依次开启
    #       "xyz" 仅平移 DoF，按 x -> y -> z 依次开启
    #       "z_rp" 三自由度，按 z -> roll -> pitch 依次开启
    #       "full" 六自由度，按 x -> y -> z -> roll -> pitch -> yaw 依次开启
    #   - dof_upgrade_every_episodes：多少轮训练后升级一个级别
    #   - amp_ramp_episodes：在多少轮内完成运动振幅从 min_amp_scale 到 1.0 的线性提升
    #   - min_amp_scale：初始振幅缩放因子
    platform_motion_levels = CurrTerm(
        func=mdp.platform_motion_levels,
        params={
            "motion_mode": "rpy",
            "dof_upgrade_every_episodes": 140,
            "amp_ramp_episodes": 800,
            "stationary_episodes": 200,
            "min_amp_scale": 0.1,
            "min_episodes_per_level": 80,
            "base_quality_threshold": 0.72,
            "per_level_quality_increment": 0.02,
            "quality_ema_alpha": 0.90,
        },
    )# 这里设置 platform_motion_levels 的 func 为 mdp.platform_motion_levels，表示使用一个基于训练进度的课程函数来逐级增加平台运动的自由度和幅度，以提供一个逐步增加挑战性的训练环境；params 中的 dof_upgrade_every_episodes 设置为 240，表示每经过 240 轮训练后升级一个平台运动自由度；amp_ramp_episodes 设置为 800，表示在升级后的 800 轮内完成平台运动振幅从 min_amp_scale 到 1.0 的线性提升；min_amp_scale 设置为 0.1，表示初始的振幅缩放因子为 10%，以提供一个较小的运动挑战，帮助训练更好地适应平台运动的挑战；如果需要更快或更慢的课程进度，可以调整 dof_upgrade_every_episodes 和 amp_ramp_episodes；如果需要更强或更弱的初始挑战，可以调整 min_amp_scale。

    # 奖励权重课程：按平台运动 level 动态调整 reward weights。
    # level=0（平台静止）保持更接近平地的姿态/高度约束，先学会走路；
    # 随着 level 提升逐步放松这些约束，减少平台运动带来的误罚。
    platform_reward_weights = CurrTerm(func=mdp.platform_reward_weight_schedule)

    # 仅用于日志：当前平台 DoF 级别（0 表示热身期静止）。
    platform_motion_level = CurrTerm(func=mdp.platform_motion_level)

    # 仅用于日志：当前平台振幅缩放系数。
    platform_amp_scale = CurrTerm(func=mdp.platform_amp_scale)
    # 仅用于日志：课程升级质量分数（即时值与EMA）。
    platform_curriculum_score = CurrTerm(func=mdp.platform_curriculum_score)
    platform_curriculum_score_ema = CurrTerm(func=mdp.platform_curriculum_score_ema)

    # 平台运动幅度课程：根据当前课程级别设置平台加速度峰值和频率。
    # params.max_linear_acc：平台线加速度上限。
    # params.lin_frequency_hz：使用正弦运动的频率。
    platform_motion_amplitude = CurrTerm(
        func=mdp.platform_motion_amplitude,
        params={
            "max_linear_acc": 0.5,
            "lin_frequency_hz": 0.2,
        },
    )# 这里设置 platform_motion_amplitude 的 func 为 mdp.platform_motion_amplitude，表示使用一个基于当前课程级别的函数来设置平台运动的加速度峰值和频率，以提供一个动态调整平台运动挑战的机制；params 中的 max_linear_acc 设置为 0.5 m/s^2，表示平台线加速度的上限；lin_frequency_hz 设置为 0.2 Hz，表示平台运动的频率；如果需要更强或更弱的运动挑战，可以调整 max_linear_acc 和 lin_frequency_hz。


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    ekf_debug_vis: bool = False
    ekf_debug_env_id: int = 0

    # Scene settings - 包含环境数量、间距等。
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=12.0)
    # Basic settings - 包含仿真时间步长、渲染间隔等。
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings - 包含奖励函数、终止条件、课程设置等。
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings - 设置仿真时间步长、渲染间隔等基本参数。
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings - 设置仿真参数。
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods 
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.left_foot_contact.update_period = self.sim.dt
        self.scene.right_foot_contact.update_period = self.sim.dt
        self.scene.base_imu.update_period = self.sim.dt
        self.scene.left_foot_imu.update_period = self.sim.dt
        self.scene.right_foot_imu.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
     # 这里在 RobotPlayEnvCfg 的 __post_init__ 方法中，首先调用了父类的 __post_init__ 方法来继承基本的环境配置，然后针对这个特定的环境进行了调整：将场景中的环境数量设置为 32，以适应更快的迭代和调试；将 base_velocity 命令的 ranges 设置为 limit_ranges，以直接使用最大命令范围，提供更大的运动挑战，帮助测试和调试算法在极限条件下的表现；如果需要更快或更慢的迭代速度，可以调整 num_envs；如果需要不同的运动挑战，可以调整 base_velocity 的 ranges。
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.ekf_debug_vis = True
        self.ekf_debug_env_id = 0
