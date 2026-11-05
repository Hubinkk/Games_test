"""
55 cm 人偶 · Isaac Lab 机器人配置（草稿）

⚠ 这份文件是在没有 Isaac Lab 的环境里写的，还没有实际运行过。
  Isaac Lab 不同版本的参数名可能有差异（例如 effort_limit / effort_limit_sim），
  在台式机上第一次使用时，请对照当前版本的文档或现有机器人配置
  （如 isaaclab_assets 里的 H1 / G1）核对一遍。

数值全部来自 sim/figure55.xml，保持两边一致：
  stiffness = MuJoCo 的 kp，damping = kv，effort_limit = forcerange
"""
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# 先用 Isaac Lab 自带的 scripts/tools/convert_urdf.py 把 figure55.urdf 转成 USD，
# 再把下面的路径改成生成的 USD 文件
FIGURE55_USD = "sim/isaac/figure55.usd"

FIGURE55_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=FIGURE55_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.29),
        joint_pos={
            ".*_hip_pitch": -0.15,
            ".*_knee": 0.3,
            ".*_ankle_pitch": -0.15,
            "l_shoulder_roll": 0.15,
            "r_shoulder_roll": -0.15,
            ".*_elbow": -0.2,
        },
    ),
    actuators={
        # 髋俯仰、膝：膝关节 v1，峰值约 2.8 N·m
        "legs_main": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch", ".*_knee"],
            effort_limit=2.8, velocity_limit=12.0, stiffness=20.0, damping=0.5,
        ),
        "hip_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll"],
            effort_limit=2.0, velocity_limit=12.0, stiffness=15.0, damping=0.4,
        ),
        "hip_yaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw"],
            effort_limit=1.0, velocity_limit=12.0, stiffness=6.0, damping=0.15,
        ),
        "ankle": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch", ".*_ankle_roll"],
            effort_limit=1.5, velocity_limit=12.0, stiffness=10.0, damping=0.25,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw", "waist_pitch"],
            effort_limit=2.0, velocity_limit=8.0, stiffness=12.0, damping=0.35,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*", ".*_elbow"],
            effort_limit=1.0, velocity_limit=10.0, stiffness=4.0, damping=0.1,
        ),
        "neck": ImplicitActuatorCfg(
            joint_names_expr=["neck_yaw", "neck_pitch"],
            effort_limit=0.3, velocity_limit=8.0, stiffness=2.0, damping=0.05,
        ),
    },
)
