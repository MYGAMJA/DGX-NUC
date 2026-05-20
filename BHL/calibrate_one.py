#!/usr/bin/env python3
"""
BHL motor 전기적 캘리브레이션 헬퍼 — 모터 1개당 1번 실행.

Usage:
    PYTHONPATH=/home/laba/Berkeley-Humanoid-Lite-Lowlevel \
        python3 /home/laba/DGX-NUC/BHL/calibrate_one.py [--can can0] [--id N]

흐름:
    1. ID sweep 1..14 (--id 주면 스킵)
    2. before config 백업 → BHL/motor_id{N}_before_cal.json
    3. calibrate_electrical_offset (20s)
    4. after config 백업 + flux_offset / phase_order diff
    5. store_settings_to_flash
    6. BHL/motor_calibration_2026-05-15.md 에 로그 추가

이미 캘리브된 모터(이전 세션) 위에 재실행하면 flux_offset이 또 갱신됨.
"""

import argparse
import datetime
import json
import math
import os
import sys
import time

import berkeley_humanoid_lite_lowlevel.recoil as recoil

BHL_DIR = "/home/laba/DGX-NUC/BHL"
LOG_MD = os.path.join(BHL_DIR, "motor_calibration_2026-05-15.md")

JOINT_MAP = {
    1:  "left_hip_roll",
    2:  "right_hip_roll",
    3:  "left_hip_yaw",
    4:  "right_hip_yaw",
    5:  "left_hip_pitch",
    6:  "right_hip_pitch",
    7:  "left_knee_pitch",
    8:  "right_knee_pitch",
    11: "left_ankle_pitch",
    12: "right_ankle_pitch",
    13: "left_ankle_roll",
    14: "right_ankle_roll",
}


def read_full_config(bus, dev):
    c = {"position_controller": {}, "current_controller": {}, "powerstage": {}, "motor": {}, "encoder": {}}
    c["device_id"] = bus._read_parameter_u32(dev, recoil.Parameter.DEVICE_ID)
    c["firmware_version"] = hex(bus._read_parameter_u32(dev, recoil.Parameter.FIRMWARE_VERSION))
    c["watchdog_timeout"] = bus._read_parameter_u32(dev, recoil.Parameter.WATCHDOG_TIMEOUT)
    c["fast_frame_frequency"] = bus.read_fast_frame_frequency(dev)
    pc = c["position_controller"]
    pc["gear_ratio"] = bus._read_parameter_f32(dev, recoil.Parameter.POSITION_CONTROLLER_GEAR_RATIO)
    pc["position_kp"] = bus.read_position_kp(dev)
    pc["position_ki"] = bus.read_position_ki(dev)
    pc["velocity_kp"] = bus.read_velocity_kp(dev)
    pc["velocity_ki"] = bus.read_velocity_ki(dev)
    pc["torque_limit"] = bus.read_torque_limit(dev)
    pc["velocity_limit"] = bus.read_velocity_limit(dev)
    pc["position_limit_upper"] = bus.read_position_limit_upper(dev)
    pc["position_limit_lower"] = bus.read_position_limit_lower(dev)
    pc["position_offset"] = bus.read_position_offset(dev)
    pc["torque_filter_alpha"] = bus.read_torque_filter_alpha(dev)
    cc = c["current_controller"]
    cc["i_limit"] = bus.read_current_limit(dev)
    cc["i_kp"] = bus.read_current_kp(dev)
    cc["i_ki"] = bus.read_current_ki(dev)
    ps = c["powerstage"]
    ps["undervoltage_threshold"] = bus._read_parameter_f32(dev, recoil.Parameter.POWERSTAGE_UNDERVOLTAGE_THRESHOLD)
    ps["overvoltage_threshold"] = bus._read_parameter_f32(dev, recoil.Parameter.POWERSTAGE_OVERVOLTAGE_THRESHOLD)
    ps["bus_voltage_filter_alpha"] = bus.read_bus_voltage_filter_alpha(dev)
    m = c["motor"]
    m["pole_pairs"] = bus.read_motor_pole_pairs(dev)
    m["torque_constant"] = bus.read_motor_torque_constant(dev)
    m["phase_order"] = bus.read_motor_phase_order(dev)
    m["max_calibration_current"] = bus.read_motor_calibration_current(dev)
    e = c["encoder"]
    e["cpr"] = bus.read_encoder_cpr(dev)
    e["position_offset"] = bus.read_encoder_position_offset(dev)
    e["velocity_filter_alpha"] = bus.read_encoder_velocity_filter_alpha(dev)
    e["flux_offset"] = bus.read_encoder_flux_offset(dev)
    return c


def sweep(bus):
    found = []
    for id_ in list(range(1, 9)) + list(range(11, 15)):
        if bus.ping(id_):
            found.append(id_)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--can", default="can0", help="CAN channel (default can0)")
    ap.add_argument("--id", type=int, default=None, help="모터 ID 지정 (skip sweep)")
    ap.add_argument("--no-store", action="store_true", help="flash 저장 생략 (테스트용)")
    args = ap.parse_args()

    now = lambda: datetime.datetime.now().strftime("%H:%M:%S")
    ts_full = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"=== [{ts_full}] calibrate_one  channel={args.can} ===")

    bus = recoil.Bus(channel=args.can, bitrate=1000000)
    try:
        # 1. ID 결정
        if args.id is not None:
            print(f"[{now()}] --id={args.id} 지정 → sweep 생략, ping 확인")
            if not bus.ping(args.id):
                print(f"ERROR: id={args.id} ping 실패. 모터 전원/CAN 확인.")
                return 1
            dev = args.id
        else:
            print(f"[{now()}] ID sweep 1..8, 11..14 ...")
            found = sweep(bus)
            if not found:
                print("ERROR: 응답하는 모터 없음. 전원/CAN 케이블 확인.")
                return 1
            if len(found) > 1:
                print(f"ERROR: 여러 모터가 응답 {found}. 한 번에 1개만 연결.")
                return 1
            dev = found[0]
        joint = JOINT_MAP.get(dev, "?")
        print(f"  → id={dev} ({joint})")

        # 2. before
        print(f"[{now()}] BEFORE config 읽는 중 ...")
        before = read_full_config(bus, dev)
        before_path = os.path.join(BHL_DIR, f"motor_id{dev}_before_cal.json")
        with open(before_path, "w") as f:
            json.dump(before, f, indent=4)
        flux_b = before['encoder']['flux_offset']
        phase_b = before['motor']['phase_order']
        cal_current = before['motor']['max_calibration_current']
        print(f"  flux_offset = {flux_b}")
        print(f"  phase_order = {phase_b}")
        print(f"  max_calibration_current = {cal_current} A")
        print(f"  → 저장: {before_path}")

        if flux_b != 0.0 and not math.isnan(flux_b):
            print(f"  ⚠️ flux_offset이 이미 0이 아님. 이 모터는 이전에 캘리브된 적 있음. 그대로 재실행함.")

        # 3. calibrate
        print(f"[{now()}] CALIBRATION 시작 — 20초 (max {cal_current} A) ...")
        bus.set_mode(dev, recoil.Mode.CALIBRATION)
        time.sleep(20)
        print(f"[{now()}] CALIBRATION 시퀀스 종료")

        # 4. after
        print(f"[{now()}] AFTER config 읽는 중 ...")
        after = read_full_config(bus, dev)
        after_path = os.path.join(BHL_DIR, f"motor_id{dev}_after_cal.json")
        with open(after_path, "w") as f:
            json.dump(after, f, indent=4)
        flux_a = after['encoder']['flux_offset']
        phase_a = after['motor']['phase_order']

        flux_str = "changed" if flux_b != flux_a else "SAME"
        phase_str = "changed" if phase_b != phase_a else "same"
        print()
        print(f"  flux_offset : {flux_b}  →  {flux_a}   ({flux_str})")
        print(f"  phase_order : {phase_b}  →  {phase_a}   ({phase_str})")

        cal_ok = (flux_b != flux_a) and (flux_a != 0.0) and not math.isnan(flux_a)
        if not cal_ok:
            print()
            print("  ⚠️ 캘리브 실패 의심 — flux_offset이 안 바뀌거나 0/NaN.")
            print("     원인 후보: 출력축 부하/회전 막힘, 전원 부족, 모터 결선.")
            print("     flash 저장 건너뜀.")
            result = "❌ 실패"
            stored = False
        else:
            # 5. store
            if args.no_store:
                print(f"\n[{now()}] --no-store 지정 → flash 저장 건너뜀")
                result = "✅ 캘리브 OK (flash 미저장)"
                stored = False
            else:
                print(f"\n[{now()}] store_settings_to_flash({dev}) ...")
                bus.store_settings_to_flash(dev)
                print(f"  → 완료. 전원 사이클 후에도 유지됨.")
                result = "✅ 캘리브 + flash 저장"
                stored = True

    finally:
        bus.stop()

    # 6. log append
    entry = f"""
### {ts_full} — id={dev} ({joint})  [calibrate_one.py]

- before: `flux_offset={flux_b}`, `phase_order={phase_b}`
- after:  `flux_offset={flux_a}`, `phase_order={phase_a}`
- 결과: {result}
- 백업: `motor_id{dev}_before_cal.json` / `motor_id{dev}_after_cal.json`
"""
    with open(LOG_MD, "a") as f:
        f.write(entry)
    print(f"\n✓ 로그 추가: {LOG_MD}")
    return 0 if (cal_ok and (stored or args.no_store)) else 1


if __name__ == "__main__":
    sys.exit(main())
