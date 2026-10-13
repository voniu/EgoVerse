# Repo Agent Rules

`CLAUDE.md` imports this file, so Claude Code and Codex follow the same rules. Keep it short.

## Read before you start
- Project knowledge (experiment registry, decision log, open questions) and machine- or
  cluster-specific instructions are kept in a private repository, not in this public fork.
  If `~/.config/agents/GLOBAL.md` exists on this machine, follow it; for model, loss, label,
  data-pipeline or training-strategy work, read `~/.config/agents/research/egoverse-piper/`
  first and do not re-propose a rejected option without new evidence.
- Architecture overview: `docs/piper_model_architecture_evolution.md`.
- Training and HPC launcher usage: `docs/piper_residual_joint_transformer_training.md`.

## Shell / Command Execution
To run commands in the interactive shell, `source emimic/bin/activate` first. Apply this before
running project Python tooling (for example `python`, `pytest`, `pip`).

## Model settings
Use plan mode for anything except extremely simple tasks.

## Tests
Tests live next to the code (`test_*.py` beside the module). Run the tests of every module you
touch; the residual joint set is listed in `docs/piper_residual_joint_transformer_training.md`
(section "Reproduction record"). Some modules fail at import for unrelated reasons
(`rldb/zarr/test_action_chunk_transforms_unit.py`, `rldb/zarr/test_zarr.py`, `algo/test_pi.py`
as of 2026-10-13); check against HEAD before attributing a failure to your change.

## Slurm rules
If you're on a Slurm cluster, request a GPU before running or testing training.
On sky1/sky2: `salloc -p rl2-lab -A rl2-lab --gres=gpu:a40:1 -c 12 --mem=30G`
On other clusters, follow the cluster instructions in the private knowledge repository.

## Write before you finish
Record results, decisions, open questions and pitfalls in the private knowledge repository
(see `~/.config/agents/GLOBAL.md`, command `/capture`), never in this public repository.
Never commit credentials, cluster accounts or unpublished results here.
