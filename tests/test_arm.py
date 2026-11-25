"""Robot-arm platform: encoders + joint torques, same blinded protocol as the AUV."""
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robotics_tools import evaluate  # noqa: E402
from robotics_tools.arm_diagnose import _ALL_PAIRS, ABSTAIN, choose_test, diagnose, simple_checks  # noqa: E402
from robotics_tools.arm_sim import ArmSecret, generate  # noqa: E402
from robotics_tools.arm_types import NOMINAL, TESTS, ArmLog, InvalidSensorLog, inverse_dynamics  # noqa: E402
from robotics_tools.diagnose import family  # noqa: E402


def _log(fam, prof="sweep", seed=0, **params):
    return generate(ArmSecret(fam, prof, seed, params))[0]


def test_log_has_no_secret_fields():
    assert {f.name for f in fields(ArmLog)} == {"enc_t", "enc_q", "tau_t", "tau"}


def test_inverse_dynamics_static_gravity():
    # arm straight out horizontally, at rest: torque is the gravity moment about each joint
    z = np.zeros((1, 2))
    m = NOMINAL
    t = inverse_dynamics(m, z, z, z)[0]
    assert np.isclose(t[1], m.m2 * m.g * m.lc2)
    assert np.isclose(t[0], (m.m1 * m.lc1 + m.m2 * (m.l1 + m.lc2)) * m.g)


def test_family_strips_axis_and_joint():
    assert family("encoder_offset_j2") == "encoder_offset" and family("mount_rotation_x") == "mount_rotation"
    assert family("timestamp_offset") == "timestamp_offset" and family("dvl_dropout") == "dvl_dropout"


@pytest.mark.parametrize("fam, params, want", [
    ("healthy", {}, "healthy"),
    ("encoder_offset", dict(joint=1, offset_rad=0.1), "encoder_offset_j1"),
    ("encoder_offset", dict(joint=2, offset_rad=0.1), "encoder_offset_j2"),
    ("timestamp_offset", dict(offset_s=0.1), "timestamp_offset"),
    ("torque_dropout", dict(start_s=5.0, duration_s=2.0), "torque_dropout"),
])
def test_dev_ranking(fam, params, want):
    d = diagnose(_log(fam, **params))
    assert d.top1 == want
    if fam == "encoder_offset":
        assert abs(d.checks["model_torque"]["offset_rad"][params["joint"] - 1] - 0.1) < 0.01
    if fam == "timestamp_offset":
        assert abs(d.checks["model_torque"]["lag_s"] - 0.1) < 0.011


def test_static_pose_abstains_then_resolves():
    d = diagnose(_log("encoder_offset", "hold", joint=1, offset_rad=0.1))
    assert d.top1 == ABSTAIN and d.next_test == choose_test(_ALL_PAIRS, {})[0]
    r = evaluate.run_case(ArmSecret("encoder_offset", "hold", 0, dict(joint=1, offset_rad=0.1)))
    assert len(r["extra_tests"]) == 1 and r["top1"] == "encoder_offset_j1"


def test_choose_test_follows_the_physics():
    # two joint offsets are told apart by moving both joints, not by a test that parks one
    t, sep = choose_test([("encoder_offset_j1", "encoder_offset_j2")], {})
    assert t in TESTS and sep > 0.3
    # a clock offset is most visible under the fastest motion
    assert choose_test([("timestamp_offset", "healthy")], {})[0] == "fast_wave"
    # a hypothesis that predicts nothing cannot be separated from healthy by any test
    assert choose_test([("torque_dropout", "healthy")], {})[1] == 0.0


def test_arm_properties_persist_and_change_the_data():
    base = _log("healthy")
    fr = generate(ArmSecret("healthy", "sweep", 0, coulomb_nm=0.2))[0]
    bl = generate(ArmSecret("healthy", "sweep", 0, backlash_rad=0.01))[0]
    assert 0.1 < np.abs(fr.tau - base.tau).max() <= 0.2 + 1e-9
    assert 0.004 < np.abs(bl.enc_q - base.enc_q).max() <= 0.005 + 1e-9


def test_baseline_cannot_name_a_clock_offset():
    assert simple_checks(_log("timestamp_offset", offset_s=0.1)) != "timestamp_offset"


def test_invalid_input():
    log = _log("healthy")
    with pytest.raises(InvalidSensorLog):
        diagnose(ArmLog(log.enc_t[:10], log.enc_q[:10], log.tau_t, log.tau))
    with pytest.raises(InvalidSensorLog):
        diagnose([])
    log.tau[5] = np.nan                                  # non-finite rows are dropped, not fatal
    assert diagnose(log).top1 == "healthy"


def test_arm_dev_split_scores():
    s = evaluate.score([evaluate.run_case(c) for c in evaluate.arm_dev_cases()])
    assert s["healthy_false_alarms"] == 0
    assert all(f["top1"] == f["n"] for f in s["per_family"].values())
