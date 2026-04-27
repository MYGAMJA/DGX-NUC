"""Hylion v4 robot configuration.

Hylion = BHL (legs + base) + SO-ARM101 × 2 (arms)
- 다리 12 joints: BHL biped와 동일하게 제어
- SO-ARM joints: stiff actuator로 기본 자세 유지 (policy에서 제외)

v4 변경: SO-ARM mount y=±0.12 (torso 충돌 박스 밖), original base_visual.stl 사용
USD 경로: δ3/usd/hylion_v4/hylion_v4/hylion_v4.usda
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

HYLION_DEFAULT_USD_PATH = "/home/laba/project_singularity/δ3/usd/hylion_v4/hylion_v4/hylion_v4.usda"
HYLION_V3_USD_PATH = os.environ.get("HYLION_USD_PATH", HYLION_DEFAULT_USD_PATH)

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

HYLION_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=HYLION_V3_USD_PATH,
        activate_contact_sensors=True,
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
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
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
        # BHL biped와 동일한 다리 actuator
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
        # SO-ARM: 높은 stiffness로 기본 자세 고정 (policy 제어 없음)
        "soarms": ImplicitActuatorCfg(
            joint_names_expr=["soarm_.*"],
            effort_limit=2.0,      # STS3215 최대 토크 ~1.86 N·m
            velocity_limit=5.0,
            stiffness=50.0,        # 팔이 흔들리지 않게 stiff
            damping=5.0,
            armature=0.001,
        ),
    },
)
