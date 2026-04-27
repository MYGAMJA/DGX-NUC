"""Hylion v6 RL 학습 스크립트 — PhysX 백엔드 (BG).

train_hylion.py 기반, Newton 코드 제거 → Isaac Lab 기본 PhysX로 동작.
gym task: Velocity-Hylion-BG-v0 (hylion_v6 USD 사용)

사용법:
  cd /home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl
  source /home/laba/env_isaaclab/bin/activate
  PYTHONUNBUFFERED=1 LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1" \\
    python /home/laba/project_singularity/δ3/scripts/train_hylion_physx_BG.py \\
      --task Velocity-Hylion-BG-v0 \\
      --num_envs 4096 \\
      --headless \\
      --max_iterations 6000
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

sys.path.insert(0, "/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl")

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="Velocity-Hylion-BG-v0")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--pretrained_checkpoint", type=str, default=None, help="Optional checkpoint path to warm-start from")
parser.add_argument(
    "--debug_obs_nan",
    action="store_true",
    default=False,
    help="Print per-term NaN counts for policy observations before learning.",
)
parser.add_argument(
    "--debug_step_nan",
    action="store_true",
    default=False,
    help="Run one env step with zero actions and print per-term NaN counts before/after step.",
)
parser.add_argument(
    "--disable_contact_sensor",
    action="store_true",
    default=False,
    help="Disable contact sensor and related reward terms (fallback for nested USD matching issues).",
)
parser.add_argument(
    "--hylion_usd_path",
    type=str,
    default=None,
    help="학습에 사용할 Hylion USD/USDA 경로. 지정 시 robot_cfg_BG 기본 경로를 override.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.hylion_usd_path:
    os.environ["HYLION_USD_PATH"] = args_cli.hylion_usd_path

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.metadata as metadata
import gymnasium as gym
import pickle
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import berkeley_humanoid_lite.tasks  # noqa: F401

# Velocity-Hylion-BG-v0 등록
sys.path.insert(0, PROJECT_ROOT)
from hylion import agents
from hylion.env_cfg_BG import HylionEnvCfg_BG

gym.register(
    id="Velocity-Hylion-BG-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": HylionEnvCfg_BG,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_ppo_cfg.HylionPPORunnerCfg,
    },
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def dump_pickle(filename: str, data: object):
    if not filename.endswith("pkl"):
        filename += ".pkl"
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(data, f)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # PhysX 기본값 사용 (NewtonCfg 미적용)

    if args_cli.disable_contact_sensor:
        print("[INFO] Disabling contact sensor and contact-dependent rewards for fallback run.")
        if hasattr(env_cfg, "scene"):
            env_cfg.scene.contact_forces = None
        if hasattr(env_cfg, "rewards"):
            if hasattr(env_cfg.rewards, "feet_air_time"):
                env_cfg.rewards.feet_air_time = None
            if hasattr(env_cfg.rewards, "feet_slide"):
                env_cfg.rewards.feet_slide = None

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    isaaclab_log_dir = os.path.join(log_root_path, "isaaclab")
    os.makedirs(isaaclab_log_dir, exist_ok=True)
    if hasattr(env_cfg.sim, "log_dir"):
        env_cfg.sim.log_dir = isaaclab_log_dir

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.debug_obs_nan:
        obs_manager = env.unwrapped.observation_manager
        old_concat = obs_manager._group_obs_concatenate.get("policy", True)
        obs_manager._group_obs_concatenate["policy"] = False
        try:
            policy_terms = obs_manager.compute_group("policy", update_history=False)
            for term_name, term_tensor in policy_terms.items():
                nan_count = int(torch.isnan(term_tensor).sum().item())
                inf_count = int(torch.isinf(term_tensor).sum().item())
                print(
                    "[OBS_DEBUG] "
                    f"term={term_name} "
                    f"shape={tuple(term_tensor.shape)} "
                    f"nan={nan_count} "
                    f"inf={inf_count}"
                )
        finally:
            obs_manager._group_obs_concatenate["policy"] = old_concat

    if args_cli.debug_step_nan:
        obs_manager = env.unwrapped.observation_manager
        old_concat = obs_manager._group_obs_concatenate.get("policy", True)
        obs_manager._group_obs_concatenate["policy"] = False
        try:
            env.reset()
            before_terms = obs_manager.compute_group("policy", update_history=False)
            for term_name, term_tensor in before_terms.items():
                nan_count = int(torch.isnan(term_tensor).sum().item())
                inf_count = int(torch.isinf(term_tensor).sum().item())
                print(
                    "[STEP_DEBUG][before] "
                    f"term={term_name} "
                    f"shape={tuple(term_tensor.shape)} "
                    f"nan={nan_count} "
                    f"inf={inf_count}"
                )

            action_dim = env.unwrapped.action_manager.total_action_dim
            zero_action = torch.zeros((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device)
            env.step(zero_action)

            after_terms = obs_manager.compute_group("policy", update_history=False)
            for term_name, term_tensor in after_terms.items():
                nan_count = int(torch.isnan(term_tensor).sum().item())
                inf_count = int(torch.isinf(term_tensor).sum().item())
                print(
                    "[STEP_DEBUG][after] "
                    f"term={term_name} "
                    f"shape={tuple(term_tensor.shape)} "
                    f"nan={nan_count} "
                    f"inf={inf_count}"
                )
        finally:
            obs_manager._group_obs_concatenate["policy"] = old_concat
            env.close()
        return

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    if args_cli.pretrained_checkpoint:
        print(f"[INFO]: Loading pretrained checkpoint from: {args_cli.pretrained_checkpoint}")
        runner.load(args_cli.pretrained_checkpoint)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
