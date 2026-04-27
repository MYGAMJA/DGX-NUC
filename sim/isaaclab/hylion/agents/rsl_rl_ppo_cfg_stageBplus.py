"""PPO config for Stage B+ (stable base policy consolidation).

Stage B+ 목표: Stage B best.pt에서 출발해 안정적인 걸음걸이를 견고히 굳힘.
- 외력/base_mass 없는 클린 환경
- 보수적 PPO: LR 낮음, epochs 2, clip 작음
- feet_air_time 1.5 (과도한 3.0 제거)
- 3000 iter 단기 fine-tuning
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class HylionPPORunnerCfg_StageBplus(RslRlOnPolicyRunnerCfg):
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
        clip_param=0.10,             # Stage B 원본 유지 (policy 변화 최소화)
        entropy_coef=0.005,          # Stage B 원본 유지
        num_learning_epochs=2,       # Stage B 원본 유지
        num_mini_batches=4,
        learning_rate=5.0e-5,        # Stage B 원본 유지 (낮게 고정)
        schedule="fixed",            # fixed: Stage B에서 배운 것 망치지 않게
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=0.05,          # Stage B 원본 유지
    )
