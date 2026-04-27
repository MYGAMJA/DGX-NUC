"""Hylion v6 robot configuration (BG).

robot_cfg.py 기반, USD 경로만 hylion_v6으로 교체.
USD 경로: δ3/usd/hylion_v6/hylion_v6/hylion_v6.usda
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Use the regenerated v6 asset under δ1 & ε2. This path was validated to produce
# non-zero foot contact forces and non-zero feet_air_time in PhysX training.
HYLION_DEFAULT_USD_PATH = "/home/laba/project_singularity/δ1 & ε2/usd/hylion_v6/hylion_v6.usda"
HYLION_USD_PATH = os.environ.get("HYLION_USD_PATH", HYLION_DEFAULT_USD_PATH)

# BHL biped와 동일한 다리 joint 목록
HYLION_LEG_JOINTS = [
    "leg_left_hip_roll_joint",
    "leg_left_hip_yaw_joint",
    "leg_left_hip_pitch_joint",
    "leg_left_knee_pitch_joint",
    "leg_left_ankle_pitch_joint",
    "leg_left_ankle_roll_joint",
    "leg_right_hip_roll_joint",
    "leg_right_hip_yaw_joint",
    "leg_right_hip_pitch_joint",
    "leg_right_knee_pitch_joint",
    "leg_right_ankle_pitch_joint",
    "leg_right_ankle_roll_joint",
]

HYLION_CFG_BG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=HYLION_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.78),
        joint_pos={
            # 다리: BHL biped 기본 자세
            "leg_left_hip_roll_joint": 0.0,
            "leg_left_hip_yaw_joint": 0.0,
            "leg_left_hip_pitch_joint": -0.2,
            "leg_left_knee_pitch_joint": 0.4,
            "leg_left_ankle_pitch_joint": -0.3,
            "leg_left_ankle_roll_joint": 0.0,
            "leg_right_hip_roll_joint": 0.0,
            "leg_right_hip_yaw_joint": 0.0,
            "leg_right_hip_pitch_joint": -0.2,
            "leg_right_knee_pitch_joint": 0.4,
            "leg_right_ankle_pitch_joint": -0.3,
            "leg_right_ankle_roll_joint": 0.0,
            # SO-ARM: 모두 기본 자세(0)
            "soarm_left_shoulder_pan": 0.0,
            "soarm_left_shoulder_lift": 0.0,
            "soarm_left_elbow_flex": 0.0,
            "soarm_left_wrist_flex": 0.0,
            "soarm_left_wrist_roll": 0.0,
            "soarm_left_gripper": 0.0,
            "soarm_right_shoulder_pan": 0.0,
            "soarm_right_shoulder_lift": 0.0,
            "soarm_right_elbow_flex": 0.0,
            "soarm_right_wrist_flex": 0.0,
            "soarm_right_wrist_roll": 0.0,
            "soarm_right_gripper": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "leg_.*_hip_yaw_joint",
                "leg_.*_hip_roll_joint",
                "leg_.*_hip_pitch_joint",
                "leg_.*_knee_pitch_joint",
            ],
            effort_limit=6,
            velocity_limit=10.0,
            stiffness=20,
            damping=2,
            armature=0.007,
        ),
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=[
                "leg_.*_ankle_pitch_joint",
                "leg_.*_ankle_roll_joint",
            ],
            effort_limit=6,
            velocity_limit=10.0,
            stiffness=20,
            damping=2,
            armature=0.002,
        ),
        "soarms": ImplicitActuatorCfg(
            joint_names_expr=["soarm_.*"],
            effort_limit=2.0,
            velocity_limit=5.0,
            stiffness=50.0,
            damping=5.0,
            armature=0.001,
        ),
    },
)
