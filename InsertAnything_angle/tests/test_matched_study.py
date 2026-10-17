"""Validate publication layout and dry-run behavior without starting Isaac Sim."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def publication_layout(tmp_path):
    project = tmp_path / "insert" / "InsertAnything_angle"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "scripts/run_matched_reward_study.py"
    shutil.copy2(source, scripts / source.name)
    shutil.copy2(source.with_name("python.sh"), scripts / "python.sh")
    base = project.parent / "source/InsertAnything/InsertAnything/tasks/direct/insertanything"
    (base / "agents").mkdir(parents=True)
    (base / "insertanything_env.py").write_text("# Base source snapshot sentinel.\n")
    (base / "agents/rl_games_ppo_L_cfg.yaml").write_text("params: {}\n")
    env = os.environ.copy()
    env.pop("INSERTANYTHING_ROOT", None)
    return project, base, env


def invoke(layout, *arguments):
    project, _, env = layout
    return subprocess.run(
        [sys.executable, str(project / "scripts/run_matched_reward_study.py"),
         "--study_name", "plan", "--plan_only", *arguments],
        env=env, capture_output=True, text=True, timeout=30,
    )


def test_plan_tracks_base_sources_and_only_changes_reward(publication_layout):
    project, base, _ = publication_layout
    result = invoke(publication_layout)
    assert result.returncode == 0, result.stderr
    state = json.loads((project / "outputs/plan/study.json").read_text())
    assert state["status"] == "planned"
    assert "child_pid" not in state
    assert not (project / "logs").exists()
    assert str(base / "insertanything_env.py") in state["source_sha256"]
    assert str(base / "agents/rl_games_ppo_L_cfg.yaml") in state["source_sha256"]
    assert state["arguments"]["lr_schedule"] == "adaptive"
    assert state["arguments"]["kl_threshold"] == 0.008
    commands = []
    for label, weight in (("control", "0.0"), ("aligned", "1.0")):
        cmd = state["arms"][label]["command"]
        assert cmd[cmd.index("--align_weight") + 1] == weight
        assert cmd[cmd.index("--lr_schedule") + 1] == "adaptive"
        assert cmd[cmd.index("--kl_threshold") + 1] == "0.008"
        assert "--depth_axis_gate" not in cmd
        for option in ("--align_weight", "--run_name"):
            cmd[cmd.index(option) + 1] = "<variant>"
        commands.append(cmd)
    assert commands[0] == commands[1]


@pytest.mark.parametrize("threshold", ["0", "nan", "inf"])
def test_plan_rejects_invalid_kl_before_creating_outputs(publication_layout, threshold):
    project, _, _ = publication_layout
    result = invoke(publication_layout, "--kl_threshold", threshold)
    assert result.returncode == 2
    assert "KL threshold" in result.stderr
    assert not (project / "outputs").exists()


def test_plan_preserves_explicit_constant_schedule(publication_layout):
    project, _, _ = publication_layout
    result = invoke(publication_layout, "--lr_schedule", "constant")
    assert result.returncode == 0, result.stderr
    state = json.loads((project / "outputs/plan/study.json").read_text())
    for arm in state["arms"].values():
        cmd = arm["command"]
        assert cmd[cmd.index("--lr_schedule") + 1] == "constant"
