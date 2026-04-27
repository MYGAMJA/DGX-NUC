"""Hylion v6 locomotion environment configuration (BG).

env_cfg.py 기반, robot을 HYLION_CFG_BG(v6)로 교체.
gym task ID: Velocity-Hylion-BG-v0
"""

import os

from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from berkeley_humanoid_lite.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityEnvCfg

from .env_cfg import CommandsCfg, ObservationsCfg, ActionsCfg, RewardsCfg, TerminationsCfg, EventsCfg, CurriculumsCfg
from .robot_cfg_BG import HYLION_CFG_BG


@configclass
class HylionEnvCfg_BG(LocomotionVelocityEnvCfg):
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculums: CurriculumsCfg = CurriculumsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 8  # 25 Hz (200 Hz physics / 8)
        self.scene.robot = HYLION_CFG_BG.replace(prim_path="{ENV_REGEX_NS}/robot")

        # Runtime overrides for matrix experiments.
        feet_air_threshold = float(os.environ.get("HYLION_FEET_AIR_THRESHOLD", "0.4"))
        leg_gain_scale = float(os.environ.get("HYLION_LEG_GAIN_SCALE", "1.0"))
        base_mass_add_kg = float(os.environ.get("HYLION_BASE_MASS_ADD_KG", "0.0"))
        contact_body_regex = os.environ.get("HYLION_CONTACT_BODY_REGEX", ".*_ankle_roll")
        # Stage-C: perturbation + wider command range
        enable_perturbation = os.environ.get("HYLION_ENABLE_PERTURBATION", "0") == "1"
        max_lin_vel_x = float(os.environ.get("HYLION_MAX_LIN_VEL_X", "0.5"))
        perturb_force = float(os.environ.get("HYLION_PERTURB_FORCE", "10.0"))
        perturb_torque = float(os.environ.get("HYLION_PERTURB_TORQUE", "3.0"))
        standing_ratio = float(os.environ.get("HYLION_STANDING_RATIO", "0.02"))

        if hasattr(self, "rewards") and hasattr(self.rewards, "feet_air_time") and self.rewards.feet_air_time is not None:
            self.rewards.feet_air_time.params["threshold"] = feet_air_threshold
            if "sensor_cfg" in self.rewards.feet_air_time.params:
                self.rewards.feet_air_time.params["sensor_cfg"].body_names = contact_body_regex
        if hasattr(self, "rewards") and hasattr(self.rewards, "feet_slide") and self.rewards.feet_slide is not None:
            if "sensor_cfg" in self.rewards.feet_slide.params:
                self.rewards.feet_slide.params["sensor_cfg"].body_names = contact_body_regex
            if "asset_cfg" in self.rewards.feet_slide.params:
                self.rewards.feet_slide.params["asset_cfg"].body_names = contact_body_regex
        # v6 안정화: 초기 디버그 단계에서는 공격적인 도메인 랜덤화/외력 주입 비활성화.
        if hasattr(self, "events"):
            if hasattr(self.events, "add_base_mass"):
                if abs(base_mass_add_kg) > 1.0e-9:
                    self.events.add_base_mass.params["mass_distribution_params"] = (base_mass_add_kg, base_mass_add_kg)
                    self.events.add_base_mass.params["operation"] = "add"
                else:
                    self.events.add_base_mass = None
            if hasattr(self.events, "base_external_force_torque"):
                if enable_perturbation:
                    # Stage-C: gentle push perturbation (작게 시작해서 curriculum 없이도 수렴 유도)
                    perturbation_force = float(os.environ.get("HYLION_PERTURB_FORCE", "5.0"))
                    perturbation_torque = float(os.environ.get("HYLION_PERTURB_TORQUE", "1.0"))
                    self.events.base_external_force_torque.params["force_range"] = (-perturbation_force, perturbation_force)
                    self.events.base_external_force_torque.params["torque_range"] = (-perturbation_torque, perturbation_torque)
                else:
                    self.events.base_external_force_torque = None
            if hasattr(self.events, "scale_all_actuator_torque_constant") and self.events.scale_all_actuator_torque_constant is not None:
                if enable_perturbation:
                    # Stage-C: randomize actuator gain ±15%
                    lo = max(0.7, leg_gain_scale - 0.15)
                    hi = leg_gain_scale + 0.15
                    self.events.scale_all_actuator_torque_constant.params["stiffness_distribution_params"] = (lo, hi)
                    self.events.scale_all_actuator_torque_constant.params["damping_distribution_params"] = (lo, hi)
                else:
                    self.events.scale_all_actuator_torque_constant.params["stiffness_distribution_params"] = (
                        leg_gain_scale,
                        leg_gain_scale,
                    )
                    self.events.scale_all_actuator_torque_constant.params["damping_distribution_params"] = (
                        leg_gain_scale,
                        leg_gain_scale,
                    )
            if hasattr(self.events, "reset_base") and self.events.reset_base is not None:
                self.events.reset_base.params["pose_range"] = {
                    "x": (-0.2, 0.2),
                    "y": (-0.2, 0.2),
                    "yaw": (-0.4, 0.4),
                }
                self.events.reset_base.params["velocity_range"] = {
                    "x": (-0.15, 0.15),
                    "y": (-0.15, 0.15),
                    "z": (0.0, 0.0),
                    "roll": (-0.15, 0.15),
                    "pitch": (-0.15, 0.15),
                    "yaw": (-0.15, 0.15),
                }
            if hasattr(self.events, "reset_robot_joints") and self.events.reset_robot_joints is not None:
                self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)
        # Command velocity range override
        if hasattr(self, "commands") and hasattr(self.commands, "base_velocity"):
            self.commands.base_velocity.ranges.lin_vel_x = (-max_lin_vel_x, max_lin_vel_x)
            # Stage-C: 제자리 서기 비율 증가 (팔 조작 시 standing 안정화 훈련)
            if enable_perturbation:
                self.commands.base_velocity.rel_standing_envs = standing_ratio
        # Stage-C: 팔 하중 변동 시뮬레이션 (SO-ARM 움직임 → CoM 이동 모델링)
        # add_base_mass를 환경변수로 이미 처리하지 않는 경우 여기서 랜덤 범위 덮어씀
        if enable_perturbation and hasattr(self, "events") and hasattr(self.events, "add_base_mass") and self.events.add_base_mass is not None:
            if abs(base_mass_add_kg) < 1.0e-9:
                # perturbation 모드 + base_mass=0: 팔 하중 변동 범위(-0.3~1.5 kg) 적용
                self.events.add_base_mass.params["mass_distribution_params"] = (-0.3, 1.5)
                self.events.add_base_mass.params["operation"] = "add"
        # Track both left/right ankle_roll links at the end of each leg chain.
        # Keep explicit leg segment names to preserve a stable path template.
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/robot/.*",
            history_length=3,
            track_air_time=True,
        )
