"""Hylion v3 locomotion task.

이 패키지를 import하면 Isaac Lab gym에 'Velocity-Hylion-v0' 환경이 등록된다.
train_hylion.py에서 sys.path에 δ3/를 추가한 뒤 import한다.

등록된 task 목록:
  Velocity-Hylion-v0          : Stage C (현재 메인, env_cfg + ppo_cfg)
  Velocity-Hylion-BG-Bplus-v0 : Stage B+ (conservative fine-tuning)
  Velocity-Hylion-BG-C1-v0   : Stage C1 (base_mass ±0.5kg, 외력 없음)
  Velocity-Hylion-BG-C2-v0   : Stage C2 (base_mass ±0.5kg, 외력 ±3N)
  Velocity-Hylion-BG-C3-v0   : Stage C3 (base_mass ±1.0kg, 외력 ±5N)
  Velocity-Hylion-BG-C4-v0   : Stage C4 (base_mass ±1.5kg, 외력 ±10N)

  [Option A — 세밀한 외력 단계 (2026-04-17)]
  Velocity-Hylion-BG-D1-v0   : Stage D1   (base_mass ±0.5kg, 외력 ±1N)
  Velocity-Hylion-BG-D2-v0   : Stage D2   (base_mass ±0.5kg, 외력 ±2N)
  Velocity-Hylion-BG-D2p5-v0 : Stage D2.5 (base_mass ±0.5kg, 외력 ±2.5N)
  Velocity-Hylion-BG-D3-v0   : Stage D3   (base_mass ±0.5kg, 외력 ±3N)
  Velocity-Hylion-BG-D4-v0   : Stage D4   (base_mass ±1.0kg, 외력 ±5N)
  Velocity-Hylion-BG-D4p5-v0 : Stage D4.5 (base_mass ±1.0kg, 외력 ±7N)
  Velocity-Hylion-BG-D5-v0   : Stage D5   (base_mass ±1.5kg, 외력 ±10N)

  [Stage E — ±30N 최종 목표 (2026-04-22)]
  Velocity-Hylion-BG-E1-v0   : Stage E1   (base_mass ±1.5kg, 외력 ±15N)
  Velocity-Hylion-BG-E2-v0   : Stage E2   (base_mass ±2.0kg, 외력 ±20N)
  Velocity-Hylion-BG-E3-v0   : Stage E3   (base_mass ±2.0kg, 외력 ±25N)
  Velocity-Hylion-BG-E4-v0   : Stage E4   (base_mass ±2.0kg, 외력 ±30N)
"""

import gymnasium as gym

from . import agents, env_cfg
from .agents import (
    rsl_rl_ppo_cfg,
    rsl_rl_ppo_cfg_stageBplus,
    rsl_rl_ppo_cfg_stageC_progressive,
    rsl_rl_ppo_cfg_stageE,
    rsl_rl_ppo_cfg_stageD_optionA,
)

gym.register(
    id="Velocity-Hylion-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.HylionEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.HylionPPORunnerCfg,
    },
)

# Stage B+: env_cfg_BG 사용 (env var로 외력/mass 비활성화), 보수적 PPO
gym.register(
    id="Velocity-Hylion-BG-Bplus-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "hylion.env_cfg_BG:HylionEnvCfg_BG",
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg_stageBplus.HylionPPORunnerCfg_StageBplus,
    },
)

# Stage C1: base_mass ±0.5kg, 외력 없음
gym.register(
    id="Velocity-Hylion-BG-C1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "hylion.env_cfg_BG:HylionEnvCfg_BG",
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg_stageC_progressive.HylionPPORunnerCfg_StageC1,
    },
)

# Stage C2: base_mass ±0.5kg, 외력 ±3N
gym.register(
    id="Velocity-Hylion-BG-C2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "hylion.env_cfg_BG:HylionEnvCfg_BG",
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg_stageC_progressive.HylionPPORunnerCfg_StageC2,
    },
)

# Stage C3: base_mass ±1.0kg, 외력 ±5N
gym.register(
    id="Velocity-Hylion-BG-C3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "hylion.env_cfg_BG:HylionEnvCfg_BG",
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg_stageC_progressive.HylionPPORunnerCfg_StageC3,
    },
)

# Stage C4: base_mass ±1.5kg, 외력 ±10N (최종 강건성)
gym.register(
    id="Velocity-Hylion-BG-C4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "hylion.env_cfg_BG:HylionEnvCfg_BG",
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg_stageC_progressive.HylionPPORunnerCfg_StageC4,
    },
)

# ── Option A (세밀한 외력 단계, 2026-04-17) ─────────────────────────────────

# Stage D1: base_mass ±0.5kg, 외력 ±1N (최초 외력 경험)
gym.register(
    id="Velocity-Hylion-BG-D1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "hylion.env_cfg_BG:HylionEnvCfg_BG",
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg_stageD_optionA.HylionPPORunnerCfg_StageD1,
    },
)

# Stage D1.5: base_mass ±0.5kg, 외력 ±1.5N (D2 NaN 폭발 대응 — 2026-04-21 추가)
gym.register(
    id="Velocity-Hylion-BG-D1p5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "hylion.env_cfg_BG:HylionEnvCfg_BG",
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg_stageD_optionA.HylionPPORunnerCfg_StageD1_5,
    },
)

# Stage D2: base_mass ±0.5kg, 외력 ±2N
gym.register(
    id="Velocity-Hylion-BG-D2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "hylion.env_cfg_BG:HylionEnvCfg_BG",
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg_stageD_optionA.HylionPPORunnerCfg_StageD2,
    },
)

# Stage D2.5: base_mass ±0.5kg, 외력 ±2.5N (±2N→±3N 갭 완충 — 2026-04-22 추가)
gym.register(
    id="Velocity-Hylion-BG-D2p5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "hylion.env_cfg_BG:HylionEnvCfg_BG",
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg_stageD_optionA.HylionPPORunnerCfg_StageD2_5,
    },
)

# Stage D3: base_mass ±0.5kg, 외력 ±3N (C2 실패 수준 재도전, 4000iter)
gym.register(
    id="Velocity-Hylion-BG-D3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "hylion.env_cfg_BG:HylionEnvCfg_BG",
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg_stageD_optionA.HylionPPORunnerCfg_StageD3,
    },
)

# Stage D4: base_mass ±1.0kg, 외력 ±5N
gym.register(
    id="Velocity-Hylion-BG-D4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "hylion.env_cfg_BG:HylionEnvCfg_BG",
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg_stageD_optionA.HylionPPORunnerCfg_StageD4,
    },
)

# Stage D4.5: base_mass ±1.0kg, 외력 ±7N (±5N→±10N 갭 완충 — 2026-04-22 추가)
gym.register(
    id="Velocity-Hylion-BG-D4p5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "hylion.env_cfg_BG:HylionEnvCfg_BG",
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg_stageD_optionA.HylionPPORunnerCfg_StageD4_5,
    },
)

# Stage D5: base_mass ±1.5kg, 외력 ±10N (최종 강건성 목표)
gym.register(
    id="Velocity-Hylion-BG-D5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "hylion.env_cfg_BG:HylionEnvCfg_BG",
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg_stageD_optionA.HylionPPORunnerCfg_StageD5,
    },
)

# ── Stage E: ±15N → ±30N 최종 강건성 목표 (2026-04-22) ──────────────────────

# Stage E1: base_mass ±1.5kg, 외력 ±15N
gym.register(
    id="Velocity-Hylion-BG-E1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "hylion.env_cfg_BG:HylionEnvCfg_BG",
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg_stageE.HylionPPORunnerCfg_StageE1,
    },
)

# Stage E2: base_mass ±2.0kg, 외력 ±20N
gym.register(
    id="Velocity-Hylion-BG-E2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "hylion.env_cfg_BG:HylionEnvCfg_BG",
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg_stageE.HylionPPORunnerCfg_StageE2,
    },
)

# Stage E3: base_mass ±2.0kg, 외력 ±25N
gym.register(
    id="Velocity-Hylion-BG-E3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "hylion.env_cfg_BG:HylionEnvCfg_BG",
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg_stageE.HylionPPORunnerCfg_StageE3,
    },
)

# Stage E4: base_mass ±2.0kg, 외력 ±30N (최종 목표 ✨)
gym.register(
    id="Velocity-Hylion-BG-E4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "hylion.env_cfg_BG:HylionEnvCfg_BG",
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg_stageE.HylionPPORunnerCfg_StageE4,
    },
)
