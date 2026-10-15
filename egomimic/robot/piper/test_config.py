from pathlib import Path

import pytest
import yaml

from egomimic.robot.piper import config as piper_config
from egomimic.robot.piper.config import check_cameras, load_config, merge

EXAMPLE = Path(piper_config.__file__).with_name("setup.example.yaml")


def test_merge_overrides_nested_values_without_mutating_base() -> None:
    base = {"tcp": {"translation_m": [0, 0, 0.1], "parent_frame": "link6"}}

    result = merge(base, {"tcp": {"translation_m": [0, 0, 0.2]}})

    assert result == {"tcp": {"translation_m": [0, 0, 0.2], "parent_frame": "link6"}}
    assert base["tcp"]["translation_m"] == [0, 0, 0.1]


def test_load_config_without_setup_has_robot_model_only(monkeypatch) -> None:
    monkeypatch.delenv("PIPER_SETUP", raising=False)

    config = load_config()

    assert config["kinematics"]["joint_names"][0] == "joint1"
    assert "cameras" not in config
    with pytest.raises(ValueError, match="cameras.d435if.serial"):
        load_config(require_cameras=True)


def test_setup_file_from_environment_overrides_tcp(tmp_path, monkeypatch) -> None:
    setup = tmp_path / "setup.yaml"
    setup.write_text(
        yaml.safe_dump(
            {
                "cameras": {
                    "d435if": {"serial": "111", "intrinsics": {"K": [[1, 0, 0]]}},
                    "d405": {"serial": "222"},
                },
                "tcp": {"translation_m": [0.0, 0.0, 0.2]},
            }
        )
    )
    monkeypatch.setenv("PIPER_SETUP", str(setup))

    config = load_config(require_cameras=True)

    assert config["tcp"]["translation_m"] == [0.0, 0.0, 0.2]
    assert config["tcp"]["parent_frame"] == "link6"
    assert config["cameras"]["d405"]["serial"] == "222"


def test_unfilled_template_is_rejected() -> None:
    config = load_config(EXAMPLE)

    with pytest.raises(ValueError) as error:
        check_cameras(config, EXAMPLE)
    message = str(error.value)
    assert "cameras.d435if.serial" in message
    assert "cameras.d405.serial" in message
    assert "cameras.d435if.intrinsics.K" in message
