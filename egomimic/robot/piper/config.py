"""Piper configuration: robot-model constants merged with one local setup file.

`piper_l.yaml` (tracked) holds the Piper L kinematics and nominal TCP. Device- and
scene-specific values (camera serials, intrinsics, hand-eye extrinsics, fitted
corrections) live in a local setup file outside version control; see
`setup.example.yaml` and `docs/piper_setup.md`. A complete legacy calibration file
is also accepted as the setup file.
"""

import copy
import os
from pathlib import Path

import yaml

MODEL_CONFIG = Path(__file__).with_name("piper_l.yaml")
SETUP_ENV = "PIPER_SETUP"
SETUP_DOC = "docs/piper_setup.md"


def merge(base, override):
    """Recursively merge mappings; values in override win."""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def resolve_setup(path=None):
    """Explicit path, else $PIPER_SETUP, else None (robot-model constants only)."""
    if path is None and os.environ.get(SETUP_ENV):
        path = os.environ[SETUP_ENV]
    return None if path is None else Path(path).expanduser()


def load_config(setup=None, require_cameras=False):
    """Merged configuration dict; require_cameras checks the camera entries are filled."""
    config = yaml.safe_load(MODEL_CONFIG.read_text())
    path = resolve_setup(setup)
    if path is not None:
        config = merge(config, yaml.safe_load(path.read_text()) or {})
    if require_cameras:
        check_cameras(config, path)
    return config


def check_cameras(config, path=None):
    source = path or f"no setup file (pass --calibration or set {SETUP_ENV})"
    cameras = config.get("cameras") or {}
    missing = []
    for name in ("d435if", "d405"):
        serial = str((cameras.get(name) or {}).get("serial") or "")
        if not serial or serial.startswith("<"):
            missing.append(f"cameras.{name}.serial")
    intrinsics = (cameras.get("d435if") or {}).get("intrinsics") or {}
    if not intrinsics.get("K"):
        missing.append("cameras.d435if.intrinsics.K")
    if missing:
        raise ValueError(
            f"{source}: missing {', '.join(missing)}; fill a setup file following {SETUP_DOC}"
        )
