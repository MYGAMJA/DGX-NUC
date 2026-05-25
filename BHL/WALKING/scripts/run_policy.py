import time

from loop_rate_limiters import RateLimiter

from berkeley_humanoid_lite_lowlevel.robot import Humanoid
from berkeley_humanoid_lite_lowlevel.robot.humanoid import State
from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController
from berkeley_humanoid_lite_lowlevel.policy.config import Cfg


# ── 시퀀스 설정 ─────────────────────────────────────────────
# (지속 시간(초), next_state, vx, vy, vyaw)
# RL_INIT: default pose로 선형 보간 (100 스텝 = 4초 @ 25Hz)
# RL_RUNNING 진입 전 RL_INIT이 완료될 때까지 대기됨
SEQUENCE = [
    (4.0,  State.RL_INIT,    0.0,  0.0,  0.0 ),  # default pose 이동
    (5.0,  State.RL_RUNNING, 0.3,  0.0,  0.0 ),  # 앞으로 5초
    (3.0,  State.RL_RUNNING, 0.0,  0.2,  0.0 ),  # 오른쪽 3초
    (3.0,  State.RL_RUNNING, 0.0,  0.0,  0.5 ),  # 제자리 회전 3초
    (0.0,  State.IDLE,       0.0,  0.0,  0.0 ),  # 정지
]
# ─────────────────────────────────────────────────────────────


cfg = Cfg.from_arguments()

print(f"Policy frequency: {1 / cfg.policy_dt} Hz")

controller = RlController(cfg)
controller.load_policy()

rate = RateLimiter(1 / cfg.policy_dt)

robot = Humanoid()
robot.enter_damping()

obs = robot.reset()

try:
    seq_idx = 0
    step_start = time.time()

    for duration, next_state, vx, vy, vyaw in SEQUENCE:
        robot.next_state = next_state
        robot.velocity_x = vx
        robot.velocity_y = vy
        robot.velocity_yaw = vyaw

        label = {State.IDLE: "IDLE", State.RL_INIT: "RL_INIT", State.RL_RUNNING: "RL_RUNNING"}.get(next_state, str(next_state))
        print(f"[{label}] vx={vx:.2f} vy={vy:.2f} vyaw={vyaw:.2f}  ({duration:.1f}s)")

        seg_end = time.time() + duration
        while time.time() < seg_end:
            actions = controller.update(obs)
            obs = robot.step(actions)
            rate.sleep()

    print("Sequence done.")

except KeyboardInterrupt:
    pass

robot.stop()
print("Stopped.")
