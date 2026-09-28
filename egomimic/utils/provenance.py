"""Small provenance records for asynchronous training and robot trials."""

import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _git(repo: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def source_record(repo: Path) -> dict:
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=normal")
    return {
        "commit": _git(repo, "rev-parse", "HEAD"),
        "branch": _git(repo, "branch", "--show-current"),
        "dirty": bool(status) if status is not None else None,
        "status": status.splitlines() if status else [],
    }


def run_record(repo: Path) -> dict:
    slurm_keys = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_RESTART_COUNT",
    )
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_record(repo),
        "argv": sys.argv,
        "hostname": socket.gethostname(),
        "slurm": {key: os.environ[key] for key in slurm_keys if key in os.environ},
    }


def write_run_record(output: Path, repo: Path) -> dict:
    record = run_record(repo)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(output)
    return record


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
