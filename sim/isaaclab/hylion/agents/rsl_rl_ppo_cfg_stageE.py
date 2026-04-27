"""PPO 설정 — Stage E (외력 ±15N → ±30N 강건성 커리큘럼).

D5(±10N) 완료 후 ±30N 최종 목표까지의 단계별 설정.
D 단계 패턴 (lr=5e-5, steps=16, value_loss_coef=0.5) 그대로 계승.

스테이지 요약:
  E1  : base_mass ±1.5kg, 외력 ±15N  (5000iter)
  E2  : base_mass ±2.0kg, 외력 ±20N  (6000iter)
  E3  : base_mass ±2.0kg, 외력 ±25N  (6000iter)
  E4  : base_mass ±2.0kg, 외력 ±30N  (7000iter)  ← 최종 목표 ✨

NaN 방지 전략 (D 단계 교훈 계승):
  - LR=5e-5 고정 (catastrophic forgetting 방지)
  - num_steps_per_env=16 (극단 transition 분산 방지)
  - value_loss_coef=0.5 (value 함수 폭발 억제)
  - max_grad_norm=0.15

작성: 2026-04-22
"""

from rsl_rl.runners import OnPolicyRunner  # noqa: F401

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from isaaclab.utils import configclass


@configclass
class HylionPPORunnerCfg_StageE1(RslRlOnPolicyRunnerCfg):
    """Stage E1: base_mass ±1.5kg, 외력 ±15N (D5 → 30N 첫 단계)"""
    num_steps_per_env = 16         # NaN 방지
    max_iterations = 5000
    save_interval = 100
    experiment_name = "hylion"
    empirical_normalization = False
    obs_groups = {"policy": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.5,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.12,
        entropy_coef=0.008,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=5.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=0.15,
    )


@configclass
class HylionPPORunnerCfg_StageE2(RslRlOnPolicyRunnerCfg):
    """Stage E2: base_mass ±2.0kg, 외력 ±20N"""
    num_steps_per_env = 16
    max_iterations = 6000
    save_interval = 100
    experiment_name = "hylion"
    empirical_normalization = False
    obs_groups = {"policy": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.5,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.12,
        entropy_coef=0.008,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=5.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=0.15,
    )


@configclass
class HylionPPORunnerCfg_StageE3(RslRlOnPolicyRunnerCfg):
    """Stage E3: base_mass ±2.0kg, 외력 ±25N"""
    num_steps_per_env = 16
    max_iterations = 6000
    save_interval = 100
    experiment_name = "hylion"
    empirical_normalization = False
    obs_groups = {"policy": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.5,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.12,
        entropy_coef=0.008,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=5.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=0.15,
    )


@configclass
class HylionPPORunnerCfg_StageE4(RslRlOnPolicyRunnerCfg):
    """Stage E4: base_mass ±2.0kg, 외력 ±30N (최종 목표 ✨)"""
    num_steps_per_env = 16
    max_iterations = 7000
    save_interval = 100
    experiment_name = "hylion"
    empirical_normalization = False
    obs_groups = {"policy": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.5,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.12,
        entropy_coef=0.008,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=5.0e-5,
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=0.15,
    )
