from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class HylionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 6000
    save_interval = 100
    experiment_name = "hylion"
    empirical_normalization = False
    obs_groups = {
        "policy": ["policy"],
    }
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.5,          # 1.0 -> 0.5: 초기 탐험 폭 줄임 (action_std 폭발 방지)
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.15,             # Stage-C: 0.1→0.15 (새 행동 학습 허용, H2O 기준)
        entropy_coef=0.01,           # Stage-C: 0.005→0.01 (perturbation 적응 탐험 강화, BHL 기준)
        num_learning_epochs=5,       # Stage-C: 2→5 (데이터 효율 향상, BHL/RSL-RL 표준)
        num_mini_batches=4,
        learning_rate=1.0e-4,        # Stage-C: 5e-5→1e-4 (새 조건 학습 속도, H2O 기준)
        schedule="adaptive",         # Stage-C: fixed→adaptive (KL 기반 자동 LR 조정)
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,             # Stage-C: 0.005→0.01 (adaptive schedule 여유 확보)
        max_grad_norm=0.1,           # Stage-C: 0.05→0.1 (RSL-RL 표준, 학습 속도 개선)
    )
