# Piper Joint Head deployment commands

`egomimic.robot.piper.rollout_joint` runs any joint-head checkpoint (MLP, FM,
or residual joint transformer) on the Piper. Run one rollout at a time from the
repository root after activating `emimic`.

## Setup file

Robot-model constants are in `egomimic/robot/piper/piper_l.yaml`. Each physical
setup (camera serials, intrinsics, hand-eye extrinsics, optional TCP or joint
zero corrections) lives in a separate file created from
`egomimic/robot/piper/setup.example.yaml` outside the repository. Pass it with
`--calibration <setup.yaml>` or `export PIPER_SETUP=<setup.yaml>`; the full
procedure is in [piper_setup.md](piper_setup.md).

## Command template

```bash
python -m egomimic.robot.piper.rollout_joint \
  --checkpoint <checkpoint> \
  --calibration <setup.yaml> \
  --flow-seed 0 \
  --execute \
  --duration 30 \
  --execution-mode chunk \
  --chunk-size 30 \
  --speed-percent 30 \
  --control-hz 30 \
  --save-every 10 \
  --max-chunk-displacement-deg 70 \
  --max-command-step-deg 15 \
  --joint-limit-tolerance-deg 15
```

- Without `--execute` the script runs inference only and sends no commands.
- `--flow-seed` fixes FM sampling and is ignored by deterministic checkpoints.
- `--control-hz` must match the training trajectories (30 Hz).
- `--max-chunk-displacement-deg`, `--max-command-step-deg` and
  `--joint-limit-tolerance-deg` are safety gates; the values above are looser
  than the script defaults. Check `--help` before changing them.
- To start from a recorded pose, add `--initialize --initial-pose <start_pose.json>`
  (created with `egomimic.robot.piper.initial_pose`).
- Each rollout writes to a new timestamped directory under
  `.local_calibration/rollouts_joint/` (override with `--output`, never reuse
  one); it records source provenance, checkpoint step and SHA256.

When comparing checkpoints (A/B, MLP/FM), keep every flag identical and change
only `--checkpoint`.
