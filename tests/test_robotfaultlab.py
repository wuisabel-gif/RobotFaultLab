"""Deterministic tests.  Run: python3 -m pytest tests/ -q"""
import ast
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robotfaultlab import evaluate, frameforge  # noqa: E402
from robotfaultlab.diagnose import ABSTAIN, diagnose, family, simple_checks  # noqa: E402
from robotfaultlab.sim import Secret, generate  # noqa: E402
from robotfaultlab.types import SensorLog  # noqa: E402


def _log(fam, prof="surge_pulses", **params):
    return generate(Secret(fam, prof, 0, params))[0]


def test_generation_is_deterministic():
    a = _log("mount_rotation", axis="z", angle_deg=20.0)
    b = _log("mount_rotation", axis="z", angle_deg=20.0)
    for f in fields(SensorLog):
        np.testing.assert_array_equal(getattr(a, f.name), getattr(b, f.name))
    assert diagnose(a).ranking == diagnose(b).ranking


def test_sensor_log_carries_no_secrets():
    names = {f.name for f in fields(SensorLog)}
    assert not names & {"family", "label", "params", "truth", "secret", "profile"}
    src = (ROOT / "robotfaultlab" / "diagnose.py").read_text()
    tree = ast.parse(src)
    modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert not modules & {"robotfaultlab.evaluate", "evaluate"} and "Secret" not in names
    assert "Secret" not in {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


def test_dev_cases_rank_correctly():
    assert diagnose(_log("healthy")).top1 == "healthy"
    assert diagnose(_log("mount_rotation", axis="z", angle_deg=20.0)).top1 == "mount_rotation_z"
    assert diagnose(_log("mount_rotation", axis="x", angle_deg=20.0)).top1 == "mount_rotation_x"
    d = diagnose(_log("timestamp_offset", offset_s=0.2))
    assert d.top1 == "timestamp_offset" and abs(d.checks["imu_dvl"]["lag_s"] - 0.2) <= 0.02
    d = diagnose(_log("dvl_dropout", start_s=8.0, duration_s=3.0))
    assert d.top1 == "dvl_dropout" and abs(d.checks["gaps"]["longest_gap_s"] - 3.0) < 0.2


def test_hover_abstains_and_requests_surge():
    d = diagnose(_log("mount_rotation", "hover", axis="x", angle_deg=20.0))
    assert d.top1 == ABSTAIN and d.next_test == "surge_pulses"


def test_sequential_protocol_resolves_hover_case():
    res = evaluate.run_case(Secret("mount_rotation", "hover", 0, dict(axis="z", angle_deg=20.0)))
    assert res["extra_tests"] == ["surge_pulses"] and family(res["top1"]) == "mount_rotation"


def test_simple_checks_miss_small_mount_and_offset():
    assert simple_checks(_log("mount_rotation", axis="z", angle_deg=20.0)) == "healthy"
    assert simple_checks(_log("timestamp_offset", offset_s=0.2)) == "healthy"
    assert simple_checks(_log("dvl_dropout", start_s=8.0, duration_s=3.0)) == "dvl_dropout"


def test_score_denominators_include_abstentions():
    results = [
        dict(secret=Secret("healthy", "p", 0), top1="healthy", top3=["healthy"], abstained=False, extra_tests=[], baseline="healthy"),
        dict(secret=Secret("healthy", "p", 1), top1="mount_rotation_x", top3=["mount_rotation_x"], abstained=False, extra_tests=[], baseline="healthy"),
        dict(secret=Secret("mount_rotation", "p", 0), top1=ABSTAIN, top3=["healthy", "mount_rotation_x"], abstained=True, extra_tests=["surge_pulses"], baseline="healthy"),
        dict(secret=Secret("unknown", "p", 0), top1=ABSTAIN, top3=[], abstained=True, extra_tests=[], baseline="healthy"),
    ]
    s = evaluate.score(results)
    assert s["healthy_false_alarms"] == 1
    assert s["per_family"]["mount_rotation"] == dict(n=1, top1=0, top3=0, abstain=1, extra=1, base=0, ff=0, ff_n=0, ff_err=0)
    assert s["per_family"]["unknown"]["top1"] == 1 and s["total_abstain"] == 2


@pytest.mark.skipif(frameforge.binary() is None, reason="frameforge binary not built")
def test_frameforge_adapter_runs_upstream_binary():
    v = frameforge.frameforge_verdict(_log("dvl_dropout", start_s=8.0, duration_s=3.0))
    assert v["verdict"] == "dvl_dropout"
    assert any(fid == "timestamps/gaps" and sev == "WARN" for sev, fid, _ in v["findings"])
    assert frameforge.frameforge_verdict(_log("healthy"))["verdict"] == "healthy"


def test_hard_split_injections():
    from robotfaultlab.types import validate
    base, _ = generate(Secret("healthy", "yaw_sweep", 20))
    nan_log, _ = generate(Secret("dvl_dropout", "yaw_sweep", 20, dict(start_s=8.0, duration_s=3.0, nan=True)))
    assert len(nan_log.dvl_t) == len(base.dvl_t) and np.isnan(nan_log.dvl_v).any()
    v = validate(nan_log)
    assert np.diff(v.dvl_t).max() > 2.9            # NaN window becomes a gap after validation
    drift, _ = generate(Secret("timestamp_offset", "yaw_sweep", 20, dict(drift=0.02)))
    assert np.isclose(drift.imu_t[-1] - base.imu_t[-1], 0.02 * base.imu_t[-1])
    loss, _ = generate(Secret("unknown", "yaw_sweep", 20, dict(kind="dvl_beam_loss", frac=0.3)))
    assert 0.15 < np.isnan(loss.dvl_v[:, 0]).mean() < 0.45
    gb, _ = generate(Secret("unknown", "yaw_sweep", 20, dict(kind="gyro_bias", bias_rps=0.05)))
    assert np.allclose(gb.imu_gyro[:, 0] - base.imu_gyro[:, 0], 0.05)
    trim, _ = generate(Secret("healthy", "trimmed_hover", 20))
    pitch = np.degrees(np.arctan2(trim.imu_acc[:, 0].mean(), trim.imu_acc[:, 2].mean()))
    assert 8.0 < abs(pitch) < 30.0                  # vehicle really holds a trim
    assert all(s.family in ("healthy", "mount_rotation", "timestamp_offset", "dvl_dropout", "unknown")
               for s in evaluate.hard_cases())
