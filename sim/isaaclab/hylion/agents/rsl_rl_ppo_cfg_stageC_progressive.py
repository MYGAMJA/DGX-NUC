"""PPO config for Stage C progressive variants.

Stage C1: base_mass ±0.5kg, 외력 없음 → 팔 하중 적응
Stage C2: base_mass ±0.5kg, 외력 ±3N  → 가벼운 충격 적응
Stage C3: base_mass ±1.0kg, 외력 ±5N  → 중간 충격 적응
Stage C4: base_mass ±1.5kg, 외력 ±10N → 최종 강건성

모든 Stage C에서 공통 PPO 설정:
- LR 약간 높임 (새 조건 적응) 단 catastrophic forgetting 방지 위해 5e-5 유지
- epochs 2→3 (중간 타협)
- clip_param 0.10→0.12 (소폭만 넓힘)
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class HylionPPORunnerCfg_StageC1(RslRlOnPolicyRunnerCfg):
    """Stage C1: base_mass ±0.5kg, 외력 없음"""
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "hylion"
    empirical_normalization = False
    obs_groups = {
        "policy": ["policy"],
    }
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.5,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.12,             # 소폭 증가 (새 질량 조건 적응)
        entropy_coef=0.007,          # 소폭 증가 (탐험 약간 허용)
        num_learning_epochs=3,       # 2→3 (중간 타협)
        num_mini_batches=4,
        learning_rate=5.0e-5,        # 유지 (catastrophic forgetting 방지)
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=0.05,
    )


@configclass
class HylionPPORunnerCfg_StageC2(RslRlOnPolicyRunnerCfg):
    """Stage C2: base_mass ±0.5kg, 외력 ±3N"""
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "hylion"
    empirical_normalization = False
    obs_groups = {
        "policy": ["policy"],
    }
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.5,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.12,
        entropy_coef=0.008,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=6.0e-5,        # 소폭 증가 (외력 적응)
        schedule="fixed",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=0.07,
    )


@configclass
class HylionPPORunnerCfg_StageC3(RslRlOnPolicyRunnerCfg):
    """Stage C3: base_mass ±1.0kg, 외력 ±5N"""
    num_steps_per_env = 24
    max_iterations = 4000
    save_interval = 100
    experiment_name = "hylion"
    empirical_normalization = False
    obs_groups = {
        "policy": ["policy"],
    }
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.5,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.13,
        entropy_coef=0.009,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=7.0e-5,
        schedule="adaptive",         # 이 단계부터 adaptive 도입 (harder task)
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=0.08,
    )


@configclass
class HylionPPORunnerCfg_StageC4(RslRlOnPolicyRunnerCfg):
    """Stage C4: base_mass ±1.5kg, 외력 ±10N — 최종 강건성"""
    num_steps_per_env = 24
    max_iterations = 4000
    save_interval = 100
    experiment_name = "hylion"
    empirical_normalization = False
    obs_groups = {
        "policy": ["policy"],
    }
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.5,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.15,
        entropy_coef=0.010,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=8.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.010,
        max_grad_norm=0.10,
    )
