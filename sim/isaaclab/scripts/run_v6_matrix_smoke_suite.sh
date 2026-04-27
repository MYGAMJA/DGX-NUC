#!/usr/bin/env bash
set -euo pipefail

# Run M1-M6 with identical short horizons for fair comparison.
# Outputs:
#   - per-run logs: /tmp/hylion_v6_physx_M{N}_suite.log
#   - summary: /tmp/hylion_v6_matrix_suite_summary.txt

ITERATIONS="${1:-300}"
NUM_ENVS="${2:-512}"
SUMMARY_FILE="/tmp/hylion_v6_matrix_suite_summary.txt"
ASSET_PATH="/home/laba/project_singularity/δ1 & ε2/usd/hylion_v6/hylion_v6.usda"
CHECKPOINT="/home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl/logs/rsl_rl/biped/2026-04-06_15-27-27/model_5999.pt"

run_case() {
  local exp_id="$1"
  local mass_add_kg leg_gain_scale feet_air_threshold log_file

  case "$exp_id" in
    M1) mass_add_kg="0.0"; leg_gain_scale="1.0"; feet_air_threshold="0.2" ;;
    M2) mass_add_kg="0.0"; leg_gain_scale="1.2"; feet_air_threshold="0.2" ;;
    M3) mass_add_kg="0.5"; leg_gain_scale="1.0"; feet_air_threshold="0.2" ;;
    M4) mass_add_kg="0.5"; leg_gain_scale="1.2"; feet_air_threshold="0.2" ;;
    M5) mass_add_kg="1.0"; leg_gain_scale="1.2"; feet_air_threshold="0.2" ;;
    M6) mass_add_kg="1.5"; leg_gain_scale="1.2"; feet_air_threshold="0.2" ;;
    *) echo "Unknown experiment id: $exp_id" >&2; return 1 ;;
  esac

  log_file="/tmp/hylion_v6_physx_${exp_id}_suite.log"

  pkill -f train_hylion_physx_BG.py || true

  pushd /home/laba/Berkeley-Humanoid-Lite/scripts/rsl_rl >/dev/null
  source /home/laba/env_isaaclab/bin/activate
  unset PYTHONPATH PYTHONHOME HYLION_CONTACT_BODY_REGEX
  export PYTHONUNBUFFERED=1
  export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1"
  export HYLION_BASE_MASS_ADD_KG="$mass_add_kg"
  export HYLION_LEG_GAIN_SCALE="$leg_gain_scale"
  export HYLION_FEET_AIR_THRESHOLD="$feet_air_threshold"

  /home/laba/env_isaaclab/bin/python /home/laba/project_singularity/δ3/scripts/train_hylion_physx_BG.py \
    --task Velocity-Hylion-BG-v0 \
    --num_envs "$NUM_ENVS" \
    --headless \
    --max_iterations "$ITERATIONS" \
    --hylion_usd_path "$ASSET_PATH" \
    --pretrained_checkpoint "$CHECKPOINT" \
    > "$log_file" 2>&1
  popd >/dev/null

  python3 - <<'PY' "$exp_id" "$log_file" "$SUMMARY_FILE"
import re
import sys
from pathlib import Path

exp_id = sys.argv[1]
log_file = Path(sys.argv[2])
summary_file = Path(sys.argv[3])

patterns = {
    "iter": re.compile(r"Learning iteration\s+(\d+)/(\d+)"),
    "value": re.compile(r"Mean value loss:\s*([-+0-9.eE]+|nan)"),
    "surrogate": re.compile(r"Mean surrogate loss:\s*([-+0-9.eE]+|nan)"),
    "reward": re.compile(r"Mean reward:\s*([-+0-9.eE]+|nan|inf|-inf)"),
    "ep_len": re.compile(r"Mean episode length:\s*([-+0-9.eE]+|nan)"),
    "action_std": re.compile(r"Mean action std:\s*([-+0-9.eE]+|nan)"),
    "air": re.compile(r"Episode_Reward/feet_air_time:\s*([-+0-9.eE]+|nan)"),
}

last = {k: None for k in patterns}
has_nan = False
for line in log_file.read_text(errors="ignore").splitlines():
    low = line.lower()
    if "nan" in low:
        has_nan = True
    for key, pat in patterns.items():
        m = pat.search(line)
        if m:
            last[key] = m.group(1) if key != "iter" else f"{m.group(1)}/{m.group(2)}"

air_val = float(last["air"]) if last["air"] not in (None, "nan") else float("nan")
action_std = float(last["action_std"]) if last["action_std"] not in (None, "nan") else float("nan")
status = "PASS"
if has_nan or (last["value"] == "nan") or (last["surrogate"] == "nan"):
    status = "FAIL_NAN"
elif air_val <= 0.0:
    status = "FAIL_AIR0"
elif action_std <= 0.0:
    status = "FAIL_STD0"

header = "EXP\tITER\tVALUE\tSURR\tREWARD\tEP_LEN\tACT_STD\tAIR\tSTATUS\n"
row = f"{exp_id}\t{last['iter']}\t{last['value']}\t{last['surrogate']}\t{last['reward']}\t{last['ep_len']}\t{last['action_std']}\t{last['air']}\t{status}\n"
if not summary_file.exists():
    summary_file.write_text(header)
with summary_file.open("a") as f:
    f.write(row)
print(row, end="")
PY
}

rm -f "$SUMMARY_FILE"

for exp_id in M1 M2 M3 M4 M5 M6; do
  echo "[SUITE] running $exp_id iterations=$ITERATIONS envs=$NUM_ENVS"
  run_case "$exp_id"
done

echo
echo "[SUITE] summary saved to $SUMMARY_FILE"
cat "$SUMMARY_FILE"