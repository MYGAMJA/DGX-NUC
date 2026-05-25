# Copyright (c) 2025, The Berkeley Humanoid Lite Project Developers.
# Adapted for Hylion (biped, 12 joints, can0/can1 only)

import time

from omegaconf import OmegaConf
import numpy as np

import berkeley_humanoid_lite_lowlevel.recoil as recoil
from berkeley_humanoid_lite_lowlevel.robot.imu import SerialImu, Baudrate


class State:
    INVALID = 0
    IDLE = 1
    RL_INIT = 2
    RL_RUNNING = 3


def linear_interpolate(start: np.ndarray, end: np.ndarray, percentage: float) -> np.ndarray:
    percentage = min(max(percentage, 0.0), 1.0)
    target = start * (1. - percentage) + end * percentage
    return target


class Humanoid:
    def __init__(self):
        self.left_leg_transport = recoil.Bus("can0")
        self.right_leg_transport = recoil.Bus("can1")

        # (bus, CAN_ID, joint_name) — Hylion 배선 기준
        # can0: ID 1,3,5,7,11,13 / can1: ID 2,4,6,8,12,14
        self.joints = [
            (self.left_leg_transport,   1,  "left_hip_roll_joint"    ),  # noqa: E241
            (self.left_leg_transport,   3,  "left_hip_yaw_joint"     ),  # noqa: E241
            (self.left_leg_transport,   5,  "left_hip_pitch_joint"   ),  # noqa: E241
            (self.left_leg_transport,   7,  "left_knee_pitch_joint"  ),  # noqa: E241
            (self.left_leg_transport,   11, "left_ankle_pitch_joint" ),  # noqa: E241
            (self.left_leg_transport,   13, "left_ankle_roll_joint"  ),  # noqa: E241

            (self.right_leg_transport,  2,  "right_hip_roll_joint"   ),  # noqa: E241
            (self.right_leg_transport,  4,  "right_hip_yaw_joint"    ),  # noqa: E241
            (self.right_leg_transport,  6,  "right_hip_pitch_joint"  ),  # noqa: E241
            (self.right_leg_transport,  8,  "right_knee_pitch_joint" ),  # noqa: E241
            (self.right_leg_transport,  12, "right_ankle_pitch_joint"),  # noqa: E241
            (self.right_leg_transport,  14, "right_ankle_roll_joint" ),  # noqa: E241
        ]

        self.imu = SerialImu(baudrate=Baudrate.BAUD_460800)
        self.imu.run_forever()

        # 외부에서 직접 설정하는 속도 명령
        self.velocity_x: float = 0.0
        self.velocity_y: float = 0.0
        self.velocity_yaw: float = 0.0

        self.state = State.IDLE
        self.next_state = State.IDLE

        # D4 default pose (학습 시 사용한 값)
        self.rl_init_positions = np.array([
            0.0, 0.0, -0.2,
            0.4,
            -0.3, 0.0,
            0.0, 0.0, -0.2,
            0.4,
            -0.3, 0.0
        ], dtype=np.float32)

        # ⚠ Hylion 조립 방향 확인 후 수정 필요
        self.joint_axis_directions = np.array([
            -1, 1, -1,
            -1,
            -1, 1,
            -1, 1, 1,
            1,
            1, 1
        ], dtype=np.float32)

        self.position_offsets = np.zeros(len(self.joints), dtype=np.float32)

        self.n_lowlevel_states = 4 + 3 + 12 + 12 + 1 + 3
        self.lowlevel_states = np.zeros(self.n_lowlevel_states, dtype=np.float32)

        self.joint_velocity_target = np.zeros(len(self.joints), dtype=np.float32)
        self.joint_position_target = np.zeros(len(self.joints), dtype=np.float32)
        self.joint_position_measured = np.zeros(len(self.joints), dtype=np.float32)
        self.joint_velocity_measured = np.zeros(len(self.joints), dtype=np.float32)

        self.init_percentage = 0.0
        self.starting_positions = np.zeros_like(self.joint_position_target, dtype=np.float32)

        config_path = "calibration.yaml"
        with open(config_path, "r") as f:
            config = OmegaConf.load(f)
        position_offsets = np.array(config.get("position_offsets", None))
        assert position_offsets.shape[0] == len(self.joints)
        self.position_offsets[:] = position_offsets

    def enter_damping(self):
        self.joint_kp = np.full(len(self.joints), 20, dtype=np.float32)
        self.joint_kd = np.full(len(self.joints), 2, dtype=np.float32)
        # D4는 6 Nm으로 학습됨 — BHL 기본값(4 Nm)보다 높게 설정
        self.torque_limit = np.full(len(self.joints), 6, dtype=np.float32)

        for i, entry in enumerate(self.joints):
            bus, device_id, joint_name = entry

            print(f"Initializing joint {joint_name}:")
            print(f"  kp: {self.joint_kp[i]}, kd: {self.joint_kd[i]}, torque limit: {self.torque_limit[i]}")

            bus.set_mode(device_id, recoil.Mode.IDLE)
            time.sleep(0.001)
            bus.write_position_kp(device_id, self.joint_kp[i])
            time.sleep(0.001)
            bus.write_position_kd(device_id, self.joint_kd[i])
            time.sleep(0.001)
            bus.write_torque_limit(device_id, self.torque_limit[i])
            time.sleep(0.001)
            bus.feed(device_id)
            bus.set_mode(device_id, recoil.Mode.DAMPING)

        print("Motors enabled")

    def stop(self):
        self.imu.stop()

        for entry in self.joints:
            bus, device_id, _ = entry
            bus.set_mode(device_id, recoil.Mode.DAMPING)

        print("Entered damping mode. Press Ctrl+C again to exit.\n")

        try:
            while True:
                pass
        except KeyboardInterrupt:
            print("Exiting damping mode.")

        for entry in self.joints:
            bus, device_id, _ = entry
            bus.set_mode(device_id, recoil.Mode.IDLE)

        self.left_leg_transport.stop()
        self.right_leg_transport.stop()

    def get_observations(self) -> np.ndarray:
        imu_quaternion = self.lowlevel_states[0:4]
        imu_angular_velocity = self.lowlevel_states[4:7]
        joint_positions = self.lowlevel_states[7:19]
        joint_velocities = self.lowlevel_states[19:31]
        mode = self.lowlevel_states[31:32]
        velocity_commands = self.lowlevel_states[32:35]

        imu_quaternion[:] = self.imu.quaternion[:]
        imu_angular_velocity[:] = np.deg2rad(self.imu.angular_velocity[:])

        joint_positions[:] = self.joint_position_measured[:]
        joint_velocities[:] = self.joint_velocity_measured[:]

        mode[0] = float(self.next_state)
        velocity_commands[0] = self.velocity_x
        velocity_commands[1] = self.velocity_y
        velocity_commands[2] = self.velocity_yaw

        return self.lowlevel_states

    def update_joint_group(self, joint_id_l, joint_id_r):
        position_target_l = (self.joint_position_target[joint_id_l] + self.position_offsets[joint_id_l]) * self.joint_axis_directions[joint_id_l]
        position_target_r = (self.joint_position_target[joint_id_r] + self.position_offsets[joint_id_r]) * self.joint_axis_directions[joint_id_r]

        self.joints[joint_id_l][0].transmit_pdo_2(self.joints[joint_id_l][1], position_target=position_target_l, velocity_target=0.0)
        self.joints[joint_id_r][0].transmit_pdo_2(self.joints[joint_id_r][1], position_target=position_target_r, velocity_target=0.0)

        position_measured_l, velocity_measured_l = self.joints[joint_id_l][0].receive_pdo_2(self.joints[joint_id_l][1])
        position_measured_r, velocity_measured_r = self.joints[joint_id_r][0].receive_pdo_2(self.joints[joint_id_r][1])

        if position_measured_l is not None:
            self.joint_position_measured[joint_id_l] = (position_measured_l * self.joint_axis_directions[joint_id_l]) - self.position_offsets[joint_id_l]
        if velocity_measured_l is not None:
            self.joint_velocity_measured[joint_id_l] = velocity_measured_l * self.joint_axis_directions[joint_id_l]
        if position_measured_r is not None:
            self.joint_position_measured[joint_id_r] = (position_measured_r * self.joint_axis_directions[joint_id_r]) - self.position_offsets[joint_id_r]
        if velocity_measured_r is not None:
            self.joint_velocity_measured[joint_id_r] = velocity_measured_r * self.joint_axis_directions[joint_id_r]

    def update_joints(self):
        self.update_joint_group(0, 6)
        self.update_joint_group(1, 7)
        self.update_joint_group(2, 8)
        self.update_joint_group(3, 9)
        self.update_joint_group(4, 10)
        self.update_joint_group(5, 11)

    def reset(self):
        return self.get_observations()

    def step(self, actions: np.ndarray):
        match (self.state):
            case State.IDLE:
                self.joint_position_target[:] = self.joint_position_measured[:]

                if self.next_state == State.RL_INIT:
                    print("Switching to RL initialization mode")
                    self.state = self.next_state

                    for entry in self.joints:
                        bus, device_id, _ = entry
                        bus.feed(device_id)
                        bus.set_mode(device_id, recoil.Mode.POSITION)

                    self.starting_positions = self.joint_position_target[:]
                    self.init_percentage = 0.0

            case State.RL_INIT:
                print(f"init: {self.init_percentage:.2f}")
                if self.init_percentage < 1.0:
                    self.init_percentage += 1 / 100.0
                    self.init_percentage = min(self.init_percentage, 1.0)
                    self.joint_position_target = linear_interpolate(self.starting_positions, self.rl_init_positions, self.init_percentage)
                else:
                    if self.next_state == State.RL_RUNNING:
                        print("Switching to RL running mode")
                        self.state = self.next_state

                    if self.next_state == State.IDLE:
                        print("Switching to idle mode")
                        self.state = self.next_state

                        for entry in self.joints:
                            bus, device_id, _ = entry
                            bus.set_mode(device_id, recoil.Mode.DAMPING)

            case State.RL_RUNNING:
                for i in range(len(self.joints)):
                    self.joint_position_target[i] = actions[i]

                if self.next_state == State.IDLE:
                    print("Switching to idle mode")
                    self.state = self.next_state

                    for entry in self.joints:
                        bus, device_id, _ = entry
                        bus.set_mode(device_id, recoil.Mode.DAMPING)

        self.update_joints()
        return self.get_observations()

    def check_connection(self):
        for entry in self.joints:
            bus, device_id, joint_name = entry
            print(f"Pinging {joint_name} ... ", end="\t")
            result = bus.ping(device_id)
            print("OK" if result else "ERROR")
            time.sleep(0.1)
