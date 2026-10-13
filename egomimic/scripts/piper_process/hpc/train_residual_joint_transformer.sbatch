#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# Job notifications (resources allocated, training started, failed, finished)
# through PIPER_NOTIFY, default ~/.config/hpc/notify.sh; silently skipped if absent.
NOTIFY=${PIPER_NOTIFY:-$HOME/.config/hpc/notify.sh}
notify() { if [[ -x "$NOTIFY" ]]; then "$NOTIFY" "$@" || true; fi; }
job_log() {
  scontrol show job -o "${SLURM_JOB_ID:-0}" 2>/dev/null | grep -oP 'StdOut=\K\S+'
}
# Append one JSON line per job to the run ledger (PIPER_LEDGER, default
# $WORK_ROOT/ledger.jsonl): the raw record that docs/research/EXPERIMENTS.md
# entries are built from. Never fails the job.
record_run() {
  local outcome=$1
  LEDGER_OUTCOME=$outcome LEDGER_RUN=${RUN:-} LEDGER_GROUP=${GROUP:-} \
  LEDGER_CONFIG=${CONFIG:-} LEDGER_CODE=${CODE:-} \
  LEDGER_OVERRIDES=$(printf '%s\n' "${OVERRIDES[@]:-}") \
  python3 - "${LEDGER:-}" <<'PY' 2>/dev/null || true
import csv, datetime, glob, json, os, subprocess, sys
env = os.environ.get
ledger, run, code = sys.argv[1], env("LEDGER_RUN"), env("LEDGER_CODE")
if not ledger:
    sys.exit()
def git(*args):
    try:
        return subprocess.run(["git", "-C", code, *args], capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""
record = {
    "time": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    "job_id": env("SLURM_JOB_ID"), "job_name": env("SLURM_JOB_NAME"),
    "node": env("SLURMD_NODENAME"), "outcome": env("LEDGER_OUTCOME"),
    "run_dir": run, "group": env("LEDGER_GROUP"), "config": env("LEDGER_CONFIG"),
    "commit": git("rev-parse", "HEAD"),
    "dirty_files": len([l for l in git("status", "--porcelain", "--untracked-files=no").splitlines() if l]),
    "overrides": [o for o in env("LEDGER_OVERRIDES", "").splitlines() if o],
}
for stage in ("train", "smoke"):
    files = sorted(glob.glob(f"{run}/{stage}/metrics/*/metrics.csv"))
    if not files:
        continue
    rows = list(csv.DictReader(open(files[-1])))
    keys = [k for k in rows[0] if k.startswith("Valid/") and k.endswith("_loss")]
    record["metrics_stage"] = stage
    record["last_step"] = max((int(r["step"]) for r in rows if r.get("step")), default=None)
    record["valid"] = {k: next((float(r[k]) for r in reversed(rows) if r.get(k)), None) for k in keys}
    record["checkpoints"] = sorted(os.path.basename(p) for p in glob.glob(f"{run}/{stage}/checkpoints/*.ckpt"))
    break
with open(ledger, "a") as handle:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
PY
}
on_exit() {
  local status=$?
  if [[ -n "${CLEAN_DIR:-}" ]]; then rm -rf "$CLEAN_DIR"; fi
  if ((status != 0)) && [[ -z "${TERMINATED:-}" ]]; then
    record_run "failed exit=$status"
    notify "报错结束 exit=$status" "run: ${RUN:-?}" \
      "$(tail -n 6 "$(job_log)" 2>/dev/null | cut -c1-200)"
  fi
}
trap on_exit EXIT
trap 'TERMINATED=1; record_run terminated; notify "收到终止信号（超时或被取消）" "run: ${RUN:-?}"; exit 143' TERM

GROUP=${1:?Pass a or b}
shift
EXTRA_OVERRIDES=("$@")
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../../.." && pwd)
CODE=${PIPER_CODE:-$REPO_ROOT}
WORK_ROOT=${PIPER_WORK_ROOT:-$CODE/.local_calibration/hpc_residual_joint_transformer}
LEDGER=${PIPER_LEDGER:-$WORK_ROOT/ledger.jsonl}
PIPER_DATA_ARCHIVE=${PIPER_DATA_ARCHIVE:-$WORK_ROOT/data/piper_zarr.tar}
ARIA_DATA_ARCHIVE=${ARIA_DATA_ARCHIVE:-$WORK_ROOT/data/aria_left_aligned.tar}
case "$GROUP" in
  a) CONFIG=train_piper_a_residual_joint_transformer ;;
  b) CONFIG=train_piper_b_residual_joint_transformer ;;
  *) echo "GROUP must be a or b" >&2; exit 2 ;;
esac

if [[ -n "${PIPER_ENV_SETUP:-}" ]]; then
  source "$PIPER_ENV_SETUP"
else
  source "${PIPER_VENV:-$CODE/emimic}/bin/activate"
fi
cd "$CODE"
# srun does not keep the activated PATH on every cluster; pin the interpreter.
PYTHON=$(command -v python)
"$PYTHON" -c "import hydra, torch"
export PYTHONPATH="$CODE:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg
export TORCH_HOME=${TORCH_HOME:-$WORK_ROOT/assets}
SCRATCH_BASE=${SLURM_TMPDIR:-${PIPER_SCRATCH_ROOT:-/tmp}}
export TMPDIR=$SCRATCH_BASE/piper-residual-joint-transformer-$GROUP-${SLURM_JOB_ID:-manual}
export MPLCONFIGDIR="$TMPDIR/mpl"

# PIPER_DATA_DIR / ARIA_DATA_DIR point at already extracted, read-only episode
# folders shared by jobs; otherwise each job extracts the archives to node scratch.
ROBOT=${PIPER_DATA_DIR:-$TMPDIR/robot}
HUMAN=${ARIA_DATA_DIR:-$TMPDIR/human}
RUN=${PIPER_RUN_DIR:-$WORK_ROOT/runs/residual-joint-transformer-$GROUP-${SLURM_JOB_ID:-manual}}
mkdir -p "$ROBOT" "$HUMAN" "$MPLCONFIGDIR" "$RUN"
CLEAN_DIR=$TMPDIR
notify "已分配到计算资源" "run: $RUN" "config: $CONFIG" \
  "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
[[ -n "${PIPER_DATA_DIR:-}" ]] || tar -xf "$PIPER_DATA_ARCHIVE" -C "$ROBOT"
# Optional Smooth v1 of the extracted copy (commanded joints only); default off.
if [[ "${PIPER_SMOOTH:-0}" == 1 ]]; then
  if [[ -n "${PIPER_DATA_DIR:-}" ]]; then
    echo "PIPER_SMOOTH rewrites episodes in place; use PIPER_DATA_ARCHIVE" >&2
    exit 2
  fi
  "$PYTHON" -m egomimic.scripts.piper_process.smooth_robot_actions \
    --dataset "$ROBOT" \
    --report "$RUN/action_smoothing_report.json"
fi

OVERRIDES=("paths.dataset_dir=$ROBOT" "num_workers=0")
if [[ "$GROUP" == b ]]; then
  [[ -n "${ARIA_DATA_DIR:-}" ]] || tar -xf "$ARIA_DATA_ARCHIVE" -C "$HUMAN"
  OVERRIDES+=("data.train_datasets.human_left_arm.resolver.folder_path=$HUMAN")
fi
OVERRIDES+=("${EXTRA_OVERRIDES[@]}")

srun --export=ALL "$PYTHON" -c 'import os, sys, torch; print("srun_env", sys.executable, os.environ.get("TORCH_HOME"), os.environ.get("TMPDIR"), torch.cuda.is_available(), flush=True)'

# Short GPU/data/FK backward check. The formal run reuses only its normalization
# statistics and starts model/optimizer training from scratch.
srun --export=ALL "$PYTHON" -m egomimic.trainHydra \
  --config-name "$CONFIG" \
  "${OVERRIDES[@]}" \
  trainer.max_steps=20 \
  trainer.limit_train_batches=20 \
  trainer.limit_val_batches=2 \
  "hydra.run.dir=$RUN/smoke"

test -f "$RUN/smoke/norm_stats/norm_stats.json"
notify "smoke 通过，开始正式训练" "run: $RUN"
srun --export=ALL "$PYTHON" -m egomimic.trainHydra \
  --config-name "$CONFIG" \
  "${OVERRIDES[@]}" \
  "norm_stats.precomputed_norm_path=$RUN/smoke/norm_stats" \
  "hydra.run.dir=$RUN/train"

summary=$("$PYTHON" - "$RUN/train" <<'PY' 2>/dev/null || true
import csv, glob, sys
run = sys.argv[1]
rows = list(csv.DictReader(open(sorted(glob.glob(run + "/metrics/*/metrics.csv"))[-1])))
keys = [k for k in rows[0] if k.startswith("Valid/") and k.endswith("_loss")]
last = {k: next(float(r[k]) for r in reversed(rows) if r.get(k)) for k in keys}
step = max(int(r["step"]) for r in rows if r.get("step"))
print(f"step {step}; " + ", ".join(f"{k}={v:.4f}" for k, v in last.items()))
print("checkpoints: " + " ".join(sorted(p.split("/")[-1] for p in glob.glob(run + "/checkpoints/*.ckpt"))))
PY
)
record_run finished
notify "训练正常结束" "run: $RUN" "$summary"
