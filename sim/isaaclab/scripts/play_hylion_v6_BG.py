"""Hylion v6 BG 학습 결과 시각화 스크립트.

task: Velocity-Hylion-BG-v0
체크포인트: M2_restart 최종 (model_11998.pt)

사용법:
    cd /home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl
    source /home/laba/env_isaaclab/bin/activate
    DISPLAY=:1 LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1" \
        python /home/laba/project_singularity/δ3/scripts/play_hylion_v6_BG.py \
            --ckpt_path /home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl/logs/rsl_rl/hylion/2026-04-15_13-36-12/model_11998.pt \
            --num_envs 1 \
            --lin_vel_x 0.2 \
            --viz kit

기본값은 M2_restart 학습 설정과 맞춰 둔다.
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl")

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play Hylion v6 BG with trained checkpoint.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Velocity-Hylion-BG-v0")
parser.add_argument(
    "--ckpt_path",
    type=str,
    default="/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl/logs/rsl_rl/hylion/2026-04-15_13-36-12/model_11998.pt",
    help="체크포인트 경로",
)
parser.add_argument("--lin_vel_x", type=float, default=0.5, help="전진 속도 명령 (m/s)")
parser.add_argument("--lin_vel_y", type=float, default=0.0, help="횡보 속도 명령 (m/s)")
parser.add_argument("--ang_vel_z", type=float, default=0.0, help="회전 속도 명령 (rad/s)")
parser.add_argument("--max_steps", type=int, default=0, help="최대 스텝 수. 0=무한")
parser.add_argument("--base_mass_add_kg", type=float, default=0.0, help="학습과 맞출 base mass 추가값")
parser.add_argument("--leg_gain_scale", type=float, default=1.2, help="학습과 맞출 leg gain scale")
parser.add_argument("--feet_air_threshold", type=float, default=0.2, help="학습과 맞출 feet air threshold")
parser.add_argument("--stand_sec", type=float, default=2.0, help="초기 기립 안정화 시간")
parser.add_argument("--warmup_sec", type=float, default=3.0, help="목표 속도 전 워밍업 시간")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
import torch
import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg, DirectRLEnvCfg, DirectMARLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg

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


def _disable_randomization(env_cfg):
    """플레이 모드: 랜덤화 비활성화."""
    if not hasattr(env_cfg, "events"):
        return
    for term in ("add_all_joint_default_pos", "base_external_force_torque"):
        if hasattr(env_cfg.events, term):
            setattr(env_cfg.events, term, None)


def _configure_command(env_cfg, lin_vel_x, lin_vel_y, ang_vel_z):
    """고정 커맨드 설정."""
    if not (hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity")):
        return
    cmd = env_cfg.commands.base_velocity
    if hasattr(cmd, "heading_command"):
        cmd.heading_command = False
    if hasattr(cmd, "rel_standing_envs"):
        cmd.rel_standing_envs = 0.0
    if hasattr(cmd, "debug_vis"):
        cmd.debug_vis = True  # 커맨드 화살표 표시
    ranges = getattr(cmd, "ranges", None)
    if ranges is not None:
        if hasattr(ranges, "lin_vel_x"):
            ranges.lin_vel_x = (lin_vel_x, lin_vel_x)
        if hasattr(ranges, "lin_vel_y"):
            ranges.lin_vel_y = (lin_vel_y, lin_vel_y)
        if hasattr(ranges, "ang_vel_z"):
            ranges.ang_vel_z = (ang_vel_z, ang_vel_z)
        if hasattr(ranges, "heading"):
            ranges.heading = (0.0, 0.0)
    print(f"[INFO] Command: lin_vel_x={lin_vel_x}, lin_vel_y={lin_vel_y}, ang_vel_z={ang_vel_z}")


def _configure_viewer(env_cfg):
    """뷰어 카메라 위치 설정 + 리셋 랜덤화 완전 제거."""
    if hasattr(env_cfg, "viewer"):
        if hasattr(env_cfg.viewer, "eye"):
            env_cfg.viewer.eye = (3.0, 0.0, 1.6)
        if hasattr(env_cfg.viewer, "lookat"):
            env_cfg.viewer.lookat = (0.0, 0.0, 0.9)

    if hasattr(env_cfg, "events"):
        # reset_base: 위치/속도 랜덤화 제거 (랜덤 roll/pitch가 리셋 직후 넘어짐의 원인)
        if hasattr(env_cfg.events, "reset_base") and env_cfg.events.reset_base is not None:
            params = getattr(env_cfg.events.reset_base, "params", None)
            if isinstance(params, dict):
                params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
                params["velocity_range"] = {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")}

        # reset_robot_joints: 관절 초기값 랜덤화 제거 (뒤틀린 자세로 시작하는 원인)
        if hasattr(env_cfg.events, "reset_robot_joints") and env_cfg.events.reset_robot_joints is not None:
            params = getattr(env_cfg.events.reset_robot_joints, "params", None)
            if isinstance(params, dict):
                params["position_range"] = (1.0, 1.0)  # 기본 자세 그대로
                params["velocity_range"] = (0.0, 0.0)


def _handle_cfg_for_rsl_rl_v5(agent_cfg):
    """RSL-RL v5+ 형식으로 변환: policy → actor/critic."""
    cfg_dict = agent_cfg.to_dict()
    policy = cfg_dict.pop("policy", None)
    if policy is not None:
        cfg_dict["actor"] = {
            "class_name": "MLPModel",
            "hidden_dims": policy.get("actor_hidden_dims", [256, 128, 128]),
            "activation": policy.get("activation", "elu"),
            "obs_normalization": False,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": policy.get("init_noise_std", 0.5),
            },
        }
        cfg_dict["critic"] = {
            "class_name": "MLPModel",
            "hidden_dims": policy.get("critic_hidden_dims", [256, 128, 128]),
            "activation": policy.get("activation", "elu"),
            "obs_normalization": False,
        }
    obs_groups = cfg_dict.get("obs_groups") or {}
    obs_groups.setdefault("policy", ["policy"])
    obs_groups.setdefault("actor", ["policy"])
    obs_groups.setdefault("critic", ["critic"])
    cfg_dict["obs_groups"] = obs_groups
    return cfg_dict


CONTROL_HZ = 25  # 200Hz physics / 8 decimation


def _build_schedule(args_cli):
    """안전한 재생용 스케줄 생성."""
    schedule = []

    if args_cli.stand_sec > 0.0:
        schedule.append((0.0, 0.0, 0.0, args_cli.stand_sec, "기립 대기"))

    has_motion = any(abs(val) > 1.0e-6 for val in (args_cli.lin_vel_x, args_cli.lin_vel_y, args_cli.ang_vel_z))
    if args_cli.warmup_sec > 0.0 and has_motion:
        schedule.append(
            (
                0.5 * args_cli.lin_vel_x,
                0.5 * args_cli.lin_vel_y,
                0.5 * args_cli.ang_vel_z,
                args_cli.warmup_sec,
                "워밍업",
            )
        )

    schedule.append((args_cli.lin_vel_x, args_cli.lin_vel_y, args_cli.ang_vel_z, 0.0, "목표 명령"))
    return schedule


def _set_command(base_env, lin_vel_x: float, lin_vel_y: float, ang_vel_z: float):
    """커맨드 매니저 내부 버퍼를 직접 덮어써서 속도 명령 변경."""
    try:
        cmd_manager = base_env.command_manager
        term = cmd_manager._terms.get("base_velocity")
        if term is None:
            # 키 이름이 다를 경우 첫 번째 term 사용
            term = next(iter(cmd_manager._terms.values()), None)
        if term is not None and hasattr(term, "_command"):
            term._command[:, 0] = lin_vel_x
            term._command[:, 1] = lin_vel_y
            term._command[:, 2] = ang_vel_z
    except Exception as e:
        print(f"[WARN] 커맨드 설정 실패: {e}")


def main():
    os.environ["HYLION_BASE_MASS_ADD_KG"] = str(args_cli.base_mass_add_kg)
    os.environ["HYLION_LEG_GAIN_SCALE"] = str(args_cli.leg_gain_scale)
    os.environ["HYLION_FEET_AIR_THRESHOLD"] = str(args_cli.feet_air_threshold)

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True
    )
    _disable_randomization(env_cfg)
    schedule_cfg = _build_schedule(args_cli)
    first = schedule_cfg[0]
    _configure_command(env_cfg, first[0], first[1], first[2])
    _configure_viewer(env_cfg)

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg.device = args_cli.device

    checkpoint = args_cli.ckpt_path
    print(f"[INFO] Checkpoint: {checkpoint}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    rsl_rl_version = metadata.version("rsl-rl-lib")
    major = int(rsl_rl_version.split(".")[0])
    if major >= 4:
        agent_cfg_dict = _handle_cfg_for_rsl_rl_v5(agent_cfg)
    else:
        agent_cfg_dict = agent_cfg.to_dict()

    runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint, map_location=agent_cfg.device)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # 시퀀스 스텝 경계 계산
    schedule = []
    cursor = 0
    final_command = (args_cli.lin_vel_x, args_cli.lin_vel_y, args_cli.ang_vel_z, "목표 명령")
    for (vx, vy, vz, dur, label) in schedule_cfg:
        if dur <= 0.0:
            final_command = (vx, vy, vz, label)
            continue
        steps = int(dur * CONTROL_HZ)
        schedule.append((cursor, cursor + steps, vx, vy, vz, label))
        cursor += steps
    total_steps = cursor

    print("[INFO] Runtime overrides:")
    print(f"  base_mass_add_kg={args_cli.base_mass_add_kg}")
    print(f"  leg_gain_scale={args_cli.leg_gain_scale}")
    print(f"  feet_air_threshold={args_cli.feet_air_threshold}")
    print("[INFO] 데모 시퀀스:")
    for (s, e, vx, vy, vz, label) in schedule:
        print(f"  스텝 {s:4d}~{e:4d}  ({(e-s)/CONTROL_HZ:.0f}초)  {label}")
    print(f"  이후 유지: vx={final_command[0]}, vy={final_command[1]}, wz={final_command[2]}")
    print("[INFO] 창을 닫으면 즉시 종료.")

    obs = env.get_observations()
    base_env = env.unwrapped

    step_count = 0
    current_label = ""
    while simulation_app.is_running():
        active_command = final_command
        for (s, e, vx, vy, vz, label) in schedule:
            if s <= step_count < e:
                active_command = (vx, vy, vz, label)
                break

        if active_command[3] != current_label:
            print(f"[{step_count:4d}] → {active_command[3]} (vx={active_command[0]:.2f}, vy={active_command[1]:.2f}, wz={active_command[2]:.2f})")
            current_label = active_command[3]
        _set_command(base_env, active_command[0], active_command[1], active_command[2])

        with torch.inference_mode():
            actions = policy(obs)
            # env action_scale=0.25 이므로 ±1.0 클리핑 금지 (훈련과 동일하게 NaN/Inf만 방어)
            actions = torch.nan_to_num(actions, nan=0.0, posinf=10.0, neginf=-10.0)
            obs, _, _, _ = env.step(actions)

        step_count += 1
        if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
            print(f"[INFO] max_steps={args_cli.max_steps} 도달. 종료합니다.")
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
