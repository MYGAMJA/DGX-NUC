"""
calibrate_joints.py — 관절 영점 캘리브레이션 (매 전원 사이클)

실행 후 12개 관절을 기계적 한계까지 수동으로 밀어준 뒤
Enter를 누르면 calibration.yaml에 저장됩니다.
"""

import time

import numpy as np
import yaml

import berkeley_humanoid_lite_lowlevel.recoil as recoil
from berkeley_humanoid_lite_lowlevel.recoil.core import Bus


# CAN 직접 접근 (Humanoid 없이 — gamepad 의존성 없음)
left_bus  = Bus("can0")
right_bus = Bus("can1")

joints = [
    (left_bus,   1,  "left_hip_roll"    ),
    (left_bus,   3,  "left_hip_yaw"     ),
    (left_bus,   5,  "left_hip_pitch"   ),
    (left_bus,   7,  "left_knee_pitch"  ),
    (left_bus,   11, "left_ankle_pitch" ),
    (left_bus,   13, "left_ankle_roll"  ),
    (right_bus,  2,  "right_hip_roll"   ),
    (right_bus,  4,  "right_hip_yaw"    ),
    (right_bus,  6,  "right_hip_pitch"  ),
    (right_bus,  8,  "right_knee_pitch" ),
    (right_bus,  12, "right_ankle_pitch"),
    (right_bus,  14, "right_ankle_roll" ),
]

joint_axis_directions = np.array([
    -1, +1, -1,
    -1,
    -1, +1,
    -1, +1, +1,
    +1,
    +1, +1
])

ideal_values = np.array([
    np.deg2rad(-10),
    np.deg2rad(+33.75),
    np.deg2rad(+56.25),
    np.deg2rad(0),
    np.deg2rad(-45),
    np.deg2rad(-15),

    np.deg2rad(+10),
    np.deg2rad(-33.75),
    np.deg2rad(+56.25),
    np.deg2rad(0),
    np.deg2rad(-45),
    np.deg2rad(+15),
])

# 댐핑 모드로 전환 (수동으로 관절 밀 수 있게)
for bus, device_id, name in joints:
    bus.set_mode(device_id, recoil.Mode.IDLE)
    time.sleep(0.001)
    bus.write_position_kp(device_id, 0.0)
    bus.write_position_kd(device_id, 1.0)
    bus.write_torque_limit(device_id, 2.0)
    bus.set_mode(device_id, recoil.Mode.DAMPING)

print("관절이 댐핑 모드입니다. 각 관절을 기계적 한계까지 밀어주세요.")
print("준비되면 Enter를 누르세요...")

limit_readings = np.array([bus.read_position_measured(did) for bus, did, _ in joints]) * joint_axis_directions
print("초기 readings:", [f"{r:.2f}" for r in limit_readings])

try:
    while True:
        joint_readings = np.array([bus.read_position_measured(did) for bus, did, _ in joints]) * joint_axis_directions

        limit_readings[0]  = min(limit_readings[0],  joint_readings[0])
        limit_readings[1]  = max(limit_readings[1],  joint_readings[1])
        limit_readings[2]  = max(limit_readings[2],  joint_readings[2])
        limit_readings[3]  = min(limit_readings[3],  joint_readings[3])
        limit_readings[4]  = min(limit_readings[4],  joint_readings[4])
        limit_readings[5]  = min(limit_readings[5],  joint_readings[5])
        limit_readings[6]  = max(limit_readings[6],  joint_readings[6])
        limit_readings[7]  = min(limit_readings[7],  joint_readings[7])
        limit_readings[8]  = max(limit_readings[8],  joint_readings[8])
        limit_readings[9]  = min(limit_readings[9],  joint_readings[9])
        limit_readings[10] = min(limit_readings[10], joint_readings[10])
        limit_readings[11] = max(limit_readings[11], joint_readings[11])

        print("\r" + "  ".join(f"{r:+.2f}" for r in limit_readings), end="", flush=True)
        time.sleep(0.05)

except KeyboardInterrupt:
    pass

print("\n\n최종 limit readings:", [f"{r:.4f}" for r in limit_readings])

offsets = limit_readings - ideal_values
print("offsets:", [f"{o:.4f}" for o in offsets])

calibration_data = {"position_offsets": [float(o) for o in offsets]}

with open("calibration.yaml", "w") as f:
    yaml.dump(calibration_data, f)

print("calibration.yaml 저장 완료.")

for bus, device_id, _ in joints:
    bus.set_mode(device_id, recoil.Mode.IDLE)

left_bus.stop()
right_bus.stop()
