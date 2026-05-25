#!/usr/bin/env python3
"""
Apply the id=7 phase_order workaround after every BESC power cycle.

This does not command motion. It only pings the device, reads diagnostics,
writes MOTOR_PHASE_ORDER, reads it back, and exits nonzero if verification fails.
"""

import argparse
import math
import sys
import time

import berkeley_humanoid_lite_lowlevel.recoil as recoil


EXPECTED_FLUX = 86.06014251708984


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--can", default="can0")
    parser.add_argument("--id", type=int, default=7)
    parser.add_argument("--phase", type=int, default=-1)
    parser.add_argument("--expect-flux", type=float, default=EXPECTED_FLUX)
    parser.add_argument("--flux-tol", type=float, default=0.01)
    parser.add_argument("--min-vbus", type=float, default=18.0)
    args = parser.parse_args()

    bus = recoil.Bus(channel=args.can, bitrate=1000000)
    try:
        if not bus.ping(args.id):
            print(f"ERROR: id{args.id} ping failed on {args.can}")
            return 2

        device_id = bus._read_parameter_u32(args.id, recoil.Parameter.DEVICE_ID)
        firmware = hex(bus._read_parameter_u32(args.id, recoil.Parameter.FIRMWARE_VERSION))
        mode = bus._read_parameter_u32(args.id, recoil.Parameter.MODE)
        error = bus._read_parameter_u32(args.id, recoil.Parameter.ERROR)
        vbus = bus._read_parameter_f32(args.id, recoil.Parameter.POWERSTAGE_BUS_VOLTAGE_MEASURED)
        flux = bus.read_encoder_flux_offset(args.id)
        phase_before = bus.read_motor_phase_order(args.id)

        print(f"id{args.id}: DEVICE_ID={device_id} FW={firmware} MODE={mode} ERROR=0x{error:04x} VBUS={vbus:.3f}")
        print(f"id{args.id}: before PHASE_ORDER={phase_before} FLUX_OFFSET={flux}")

        if device_id != args.id:
            print(f"ERROR: responded device_id={device_id}, expected {args.id}")
            return 3
        if error != 0:
            print("ERROR: BESC error flag is nonzero; clear/diagnose before motion")
            return 4
        if not math.isfinite(vbus) or vbus < args.min_vbus:
            print(f"ERROR: VBUS too low ({vbus:.3f} V)")
            return 5
        if abs(flux - args.expect_flux) > args.flux_tol:
            print(f"ERROR: flux_offset mismatch; got {flux}, expected about {args.expect_flux}")
            return 6

        if phase_before != args.phase:
            print(f"writing PHASE_ORDER={args.phase} (temporary; rerun after power cycle)")
            bus.write_motor_phase_order(args.id, args.phase)
            time.sleep(0.2)

        phase_after = bus.read_motor_phase_order(args.id)
        error_after = bus._read_parameter_u32(args.id, recoil.Parameter.ERROR)
        print(f"id{args.id}: after PHASE_ORDER={phase_after} ERROR=0x{error_after:04x}")

        if phase_after != args.phase:
            print(f"ERROR: phase_order verify failed; got {phase_after}, expected {args.phase}")
            return 7
        if error_after != 0:
            print("ERROR: BESC error flag became nonzero after write")
            return 8

        print("OK: id7 phase_order workaround is applied; do not power-cycle before motion test")
        return 0
    finally:
        bus.stop()


if __name__ == "__main__":
    sys.exit(main())
