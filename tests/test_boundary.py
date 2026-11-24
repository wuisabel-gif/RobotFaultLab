"""Blinding boundary, relabeling invariance, invalid input, and adapter failure tests."""
import dataclasses
import os
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robotfaultlab import frameforge  # noqa: E402
from robotfaultlab.diagnose import diagnose, simple_checks  # noqa: E402
from robotfaultlab.sim import Secret, generate  # noqa: E402
from robotfaultlab.types import InvalidSensorLog, SensorLog, validate  # noqa: E402


def _log(fam, prof="surge_pulses", seed=0, **params):
    return generate(Secret(fam, prof, seed, params))[0]


# ---- process-level import boundary -------------------------------------------------
@pytest.mark.parametrize("module", ["robotfaultlab.types", "robotfaultlab.diagnose", "robotfaultlab.frameforge", "robotfaultlab.bag",
                                    "robotfaultlab.arm_types", "robotfaultlab.arm_diagnose"])
def test_diagnostic_side_imports_no_simulator_or_evaluator(module):
    code = (
        "import sys, importlib; sys.path.insert(0, %r); importlib.import_module(%r); "
        "bad = sorted(m for m in sys.modules if m.startswith(('robotfaultlab.sim', 'robotfaultlab.arm_sim', 'robotfaultlab.evaluate', 'auv_sim_bench'))); "
        "print(bad)" % (str(ROOT), module)
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", f"{module} pulled in: {out.stdout}"


def test_sensor_log_has_no_label_fields():
    names = {f.name for f in dataclasses.fields(SensorLog)}
    assert not names & {"family", "label", "params", "truth", "secret", "profile", "seed"}


# ---- relabeling invariance ------------------------------------------------------------
def test_identical_sensor_input_gives_identical_diagnosis_under_relabeling():
    base = _log("mount_rotation", axis="z", angle_deg=20.0)
    # The evaluator "relabels": a different Secret that happens to yield the same sensor arrays
    # is simulated by copying the arrays into a fresh SensorLog, bypassing generate().
    copy = SensorLog(**{f.name: np.array(getattr(base, f.name), copy=True) for f in dataclasses.fields(SensorLog)})
    d1, d2 = diagnose(base), diagnose(copy)
    assert d1.ranking == d2.ranking and d1.next_test == d2.next_test and d1.checks == d2.checks
    # and the wrong-label control: a different secret produces a different log, so the
    # equality above is about the sensor input, not about anything the evaluator knows.
    other = _log("timestamp_offset", offset_s=0.2)
    assert diagnose(other).ranking != d1.ranking


# ---- invalid / degenerate inputs -----------------------------------------------------
def _fields(log):
    return {f.name: np.array(getattr(log, f.name), copy=True) for f in dataclasses.fields(SensorLog)}


def test_nonfinite_rows_are_dropped_not_fatal():
    f = _fields(_log("healthy"))
    f["imu_acc"][100:120] = np.nan
    f["dvl_v"][5] = np.inf
    f["depth"][3] = np.nan
    d = diagnose(SensorLog(**f))
    assert d.top1 == "healthy"


def test_empty_and_too_short_inputs_raise():
    f = _fields(_log("healthy"))
    for key in ("imu_t", "dvl_t", "depth_t"):
        g = dict(f)
        n = 0
        for k in f:
            if k.startswith(key.split("_")[0]):
                g[k] = f[k][:n]
        with pytest.raises(InvalidSensorLog):
            diagnose(SensorLog(**g))
    g = dict(f)
    g["dvl_t"], g["dvl_v"] = f["dvl_t"][:5], f["dvl_v"][:5]
    with pytest.raises(InvalidSensorLog):
        diagnose(SensorLog(**g))
    with pytest.raises(InvalidSensorLog):
        diagnose([])


def test_all_nan_and_bad_shapes_and_bad_time_raise():
    f = _fields(_log("healthy"))
    g = dict(f); g["imu_acc"][:] = np.nan
    with pytest.raises(InvalidSensorLog):
        validate(SensorLog(**g))
    g = dict(f); g["dvl_v"] = f["dvl_v"][:, :2]
    with pytest.raises(InvalidSensorLog):
        validate(SensorLog(**g))
    g = dict(f); g["imu_t"] = f["imu_t"][::-1].copy()
    with pytest.raises(InvalidSensorLog):
        validate(SensorLog(**g))
    g = dict(f); g["imu_q"][:] = 0.0
    with pytest.raises(InvalidSensorLog):
        validate(SensorLog(**g))
    with pytest.raises(InvalidSensorLog):
        validate("not a log")


def test_zero_signal_input_abstains_without_division_error():
    f = _fields(_log("healthy"))
    for k in ("imu_acc", "imu_gyro", "dvl_v", "depth"):
        f[k][:] = 0.0
    with np.errstate(all="raise"):
        d = diagnose(SensorLog(**f))
    assert d.top1 == "abstain"
    assert simple_checks(SensorLog(**f)) == "mount_rotation"   # degenerate gravity -> FrameForge-style FAIL


# ---- FrameForge adapter failure modes ---------------------------------------------
def _fake_exe(tmp_path, body):
    p = tmp_path / "fake_frameforge"
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


GOOD_GRAVITY = 'printf "\\nPASS  [gravity/direction]  ok\\n\\n----\\nPASS  1 pass, 0 info, 0 warn, 0 fail\\n"'
GOOD_TS = ('printf "\\nPASS  [timestamps/monotonic]  ok\\n\\nWARN  [timestamps/gaps]  1 gap\\n\\n----\\n'
           'PASS  2 pass, 0 info, 1 warn, 0 fail\\n"')


@pytest.mark.parametrize("body, needle", [
    ("exit 3", "exited 3"),
    ("exit 0", "no findings"),
    ('printf "garbage\\nmore garbage\\n"; exit 0', "no findings"),
    ('printf "PASS  [gravity/direction]  ok\\n"; exit 0', "summary line"),            # no summary line
    ('printf "\\nPASS  [timestamps/monotonic]  ok\\n\\n----\\nPASS  1 pass, 0 info, 0 warn, 0 fail\\n"', "required finding missing"),
    ("sleep 5", "timed out"),
])
def test_adapter_raises_instead_of_defaulting(tmp_path, body, needle):
    exe = _fake_exe(tmp_path, body)
    with pytest.raises(frameforge.FrameForgeError, match=needle):
        frameforge.run(["gravity", "x.csv"], exe=exe, timeout=1.0)


def test_verdict_is_error_not_healthy_on_failure(tmp_path):
    v = frameforge.frameforge_verdict(_log("healthy"), exe=_fake_exe(tmp_path, "exit 3"))
    assert v["verdict"] == "error" and "exited 3" in v["error"] and v["findings"] == []
    v = frameforge.frameforge_verdict(_log("healthy"), exe=str(tmp_path / "missing"))
    assert v["verdict"] == "error"


def test_verdict_maps_fake_findings(tmp_path):
    body = f'case "$1" in gravity) {GOOD_GRAVITY};; timestamps) {GOOD_TS};; esac'
    v = frameforge.frameforge_verdict(_log("healthy"), exe=_fake_exe(tmp_path, body))
    assert v["verdict"] == "dvl_dropout"


def test_missing_binary_is_an_error(monkeypatch):
    monkeypatch.setenv("ROBOTFAULTLAB_FRAMEFORGE", "/nonexistent/frameforge")
    monkeypatch.setattr(frameforge, "binary", lambda: None)
    with pytest.raises(frameforge.FrameForgeError, match="not found"):
        frameforge.run(["gravity", "x.csv"])
