"""PPO 설정 — Stage E (외력 ±15N → ±50N 강건성 커리큘럼).

D5(±10N) 완료 후 ±50N 최종 목표까지의 단계별 설정.

옵션 A 적용 (2026-05-01) — BHL biped 정공법 채택:
  - 이전 conservative 설정 (lr=5e-5, max_grad_norm=0.15, fixed schedule, epochs=2)이
    오히려 학습 실패의 원인 (value explosion at iter 39032 in D4.5 with ±7N).
  - BHL biped 본가 설정으로 전환:
      learning_rate=1e-3, schedule=adaptive, max_grad_norm=1.0,
      num_learning_epochs=5, value_loss_coef=1.0, clip_param=0.2,
      num_steps_per_env=24, desired_kl=0.01
  - 우리 변경: entropy_coef 0.008 → 0.012 (외란 강도가 BHL보다 높으므로 탐험 강화)
  - max_iterations 4000 (오케스트레이터에서 stage별 override 가능)

옵션 B 보정 (2026-05-01, 옵션 A 1차 시도 실패 후):
  - 1차 시도 결과: D4.5 970 iter 에서 reward 11.91 → 7.52, fall 27.9% → 36.4%
                  (이전 실패와 비슷한 패턴, 재학습 가치 없음)
  - 진단: init_noise_std=1.0 이 D4 (0.5로 학습됨) actor distribution을 갑자기
          두 배로 흐트림 → 학습 초기 정책 불안정
  - 수정: init_noise_std 1.0 → 0.5 (D4와 일치)
  - 나머지 BHL 설정 (lr=1e-3, adaptive, max_grad_norm=1.0 등) 유지

CLI `--max_iterations N` 으로 stage 별 override.
외란/mass는 환경변수(HYLION_PERTURB_FORCE 등)로 제어 — 프랙셔널 stage는 task ID 재사용.

작성: 2026-04-22 | 갱신: 2026-05-01 (옵션 B 적용)
"""

from rsl_rl.runners import OnPolicyRunner  # noqa: F401

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)
from isaaclab.utils import configclass


# ── 모든 E 스테이지 공통 설정 (BHL 정공법) ─────────────────────────────────────

def _bhl_policy_cfg():
    return RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.5,   # 옵션 B (2026-05-01): D4와 일치, 1.0은 학습 초기 정책 불안정 유발
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )


def _bhl_algorithm_cfg():
    return RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.012,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class HylionPPORunnerCfg_StageE1(RslRlOnPolicyRunnerCfg):
    """Stage E1: 외력 ±15N (BHL 설정)"""
    num_steps_per_env = 24
    max_iterations = 4000
    save_interval = 100
    experiment_name = "hylion"
    empirical_normalization = False
    obs_groups = {"policy": ["policy"]}
    policy = _bhl_policy_cfg()
    algorithm = _bhl_algorithm_cfg()


@configclass
class HylionPPORunnerCfg_StageE2(RslRlOnPolicyRunnerCfg):
    """Stage E2: 외력 ±20N (BHL 설정)"""
    num_steps_per_env = 24
    max_iterations = 4000
    save_interval = 100
    experiment_name = "hylion"
    empirical_normalization = False
    obs_groups = {"policy": ["policy"]}
    policy = _bhl_policy_cfg()
    algorithm = _bhl_algorithm_cfg()


@configclass
class HylionPPORunnerCfg_StageE3(RslRlOnPolicyRunnerCfg):
    """Stage E3: 외력 ±25N (BHL 설정)"""
    num_steps_per_env = 24
    max_iterations = 4000
    save_interval = 100
    experiment_name = "hylion"
    empirical_normalization = False
    obs_groups = {"policy": ["policy"]}
    policy = _bhl_policy_cfg()
    algorithm = _bhl_algorithm_cfg()


@configclass
class HylionPPORunnerCfg_StageE4(RslRlOnPolicyRunnerCfg):
    """Stage E4: 외력 ±30N (BHL 설정)"""
    num_steps_per_env = 24
    max_iterations = 4000
    save_interval = 100
    experiment_name = "hylion"
    empirical_normalization = False
    obs_groups = {"policy": ["policy"]}
    policy = _bhl_policy_cfg()
    algorithm = _bhl_algorithm_cfg()
