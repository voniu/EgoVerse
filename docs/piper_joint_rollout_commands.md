# Piper Joint Head deployment commands

All commands use the same deployment settings. Run one command at a time from the
repository root after activating `emimic`.

## 1. A + MLP (raw actions, best step 5300)

```bash
python -m egomimic.robot.piper.rollout_joint \
  --checkpoint .local_calibration/hpc/joint/checkpoints/a/best.ckpt \
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

## 2. B + MLP (raw actions, best step 4100)

```bash
python -m egomimic.robot.piper.rollout_joint \
  --checkpoint .local_calibration/hpc/joint/checkpoints/b/best.ckpt \
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

## 3. A + FM (raw actions, best step 9600)

```bash
python -m egomimic.robot.piper.rollout_joint \
  --checkpoint .local_calibration/hpc/joint/checkpoints/a/flow_best.ckpt \
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

## 4. B + FM (raw actions, best step 9800)

```bash
python -m egomimic.robot.piper.rollout_joint \
  --checkpoint .local_calibration/hpc/joint/checkpoints/b/flow_best.ckpt \
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

## 5. A + MLP Smooth v1 (best step 6700)

```bash
python -m egomimic.robot.piper.rollout_joint \
  --checkpoint .local_calibration/hpc/joint_smooth_v1/checkpoints/a/mlp/best.ckpt \
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

## 6. A + FM Smooth v1 (best step 9600)

```bash
python -m egomimic.robot.piper.rollout_joint \
  --checkpoint .local_calibration/hpc/joint_smooth_v1/checkpoints/a/flow/best.ckpt \
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

## 7. A + FM Extended (about 17,100 cumulative updates)

```bash
python -m egomimic.robot.piper.rollout_joint \
  --checkpoint .local_calibration/hpc/joint_flow_extend/checkpoints/a/best.ckpt \
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

## 8. B + FM Extended (about 16,000 cumulative updates)

```bash
python -m egomimic.robot.piper.rollout_joint \
  --checkpoint .local_calibration/hpc/joint_flow_extend/checkpoints/b/best.ckpt \
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
