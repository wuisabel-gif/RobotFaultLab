"""Sensor-only diagnostic.  The ONLY input type is SensorLog (no labels, no truth).
This module never imports Secret or anything from evaluate.py.

Checks (each returns plain numbers; ranking is a hand-set rule table):
  gravity_axis   FrameForge-style: dominant axis of mean specific force      (simple/existing)
  dvl_gaps       FrameForge-style: DVL stamp gaps vs nominal period          (simple/existing)
  imu_dvl        NEW cross-check: joint time-lag + rotation fit between
                 IMU-derived and DVL-derived body acceleration.  A mount
                 rotation shows as a fixed rotation; a clock offset shows as
                 a lag; neither can mimic the other once both are fitted.
  depth_dvl      NEW cross-check: vertical DVL velocity (through IMU
                 orientation) vs depth rate.  Roll/pitch mounts break it,
                 yaw mounts and clock offsets do not.

Thresholds in TH were set by hand looking at the dev split only
(evaluate.DEV_CASES).  Holdout magnitudes/profiles were not consulted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .types import InvalidSensorLog, SensorLog, validate  # neutral module: no sim/evaluator imports

HYPOTHESES = ["healthy", "mount_rotation_x", "mount_rotation_y", "mount_rotation_z",
              "timestamp_offset", "dvl_dropout"]
ABSTAIN = "abstain"
TH = dict(gap_s=0.5, lag_s=0.04, rot_deg=5.0, excite_mps2=0.12, excite_strong=0.5,
          evidence_mps2=0.3, unexplained_mps2=0.2, depth_mps=0.1, close=0.2)
LAG_GRID = np.arange(-1.0, 1.0 + 1e-9, 0.01)


def family(h: str) -> str:
    return "mount_rotation" if h.startswith("mount_rotation") else h


@dataclass
class Diagnosis:
    ranking: list[tuple[str, float]]       # (hypothesis, score) sorted desc; may start with ABSTAIN
    checks: dict
    next_test: str | None = None           # motion profile to request, or None
    notes: list[str] = field(default_factory=list)

    @property
    def top1(self):
        return self.ranking[0][0]


# ---- helpers ---------------------------------------------------------------------------
def _rot_batch(q):
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], 1)


def _interp(t_src, y_src, t_dst):
    return np.stack([np.interp(t_dst, t_src, y_src[:, i]) for i in range(y_src.shape[1])], -1)


def _segments(t, factor=3.0):
    """Contiguous index ranges where dt <= factor * median dt."""
    dts = np.diff(t)
    nominal = float(np.median(dts))
    cut = np.where(dts > factor * nominal)[0]
    starts = np.r_[0, cut + 1]
    ends = np.r_[cut + 1, len(t)]
    return [(a, b) for a, b in zip(starts, ends) if b - a >= 3], nominal


def _kabsch(src, dst, anchor=None):
    """R minimising |R src - dst|; returns R, angle_deg, axis_letter.

    anchor=(src_dir, dst_dir, weight) adds a weighted direction pair.  We anchor
    with the mean specific-force direction (measured) vs body +Z (assumed level
    at rest, the same assumption FrameForge's gravity check makes); without it a
    surge-only run leaves rotations about the surge axis unobservable.
    """
    H = src.T @ dst
    if anchor is not None:
        a, b, w = anchor
        H = H + w * np.outer(a, b)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    axis = "xyz"[int(np.argmax(np.abs(ax)))] if ang > 1e-6 else "-"
    return R, float(ang), axis


def _rms(a):
    a = np.atleast_2d(a)
    return float(np.sqrt(np.mean(np.sum(a ** 2, -1)))) if a.size else 0.0


# ---- checks ----------------------------------------------------------------------------
def gravity_axis_check(log: SensorLog) -> dict:
    m = log.imu_acc.mean(0)
    dom = int(np.argmax(np.abs(m)))
    norm = float(np.linalg.norm(m))
    if norm < 1e-6:
        return dict(mean=m.round(3).tolist(), dominant="-", sign=0, tilt_deg=float("nan"), fail=True)
    tilt = float(np.degrees(np.arccos(np.clip(m[2] / norm, -1, 1))))
    return dict(mean=m.round(3).tolist(), dominant="xyz"[dom], sign=int(np.sign(m[dom])),
                tilt_deg=round(tilt, 2), fail=bool(dom != 2 or m[2] < 0))


def dvl_gap_check(log: SensorLog) -> dict:
    dts = np.diff(log.dvl_t)
    nominal = float(np.median(dts))
    gaps = dts[dts > 3 * nominal]
    return dict(nominal_dt=round(nominal, 4), n_gaps=int(len(gaps)),
                longest_gap_s=round(float(gaps.max()) if len(gaps) else 0.0, 3))


def imu_dvl_check(log: SensorLog) -> dict:
    g = float(np.linalg.norm(log.imu_acc.mean(0)))           # sensor-only gravity estimate
    if g < 1.0:
        return dict(ok=False, reason=f"mean specific force {g:.2f} m/s^2 is not gravity-like")
    R = _rot_batch(log.imu_q)
    g_imu = np.einsum("nji,j->ni", R, np.array([0, 0, g]))   # R^T g
    a_imu = log.imu_acc - g_imu                               # linear accel, IMU frame
    k = max(1, int(round(100.0 / 8.0)))                       # box filter to DVL bandwidth
    a_imu = np.stack([np.convolve(a_imu[:, i], np.ones(k) / k, "same") for i in range(3)], -1)

    segs, _ = _segments(log.dvl_t)
    idx = np.concatenate([np.arange(a + 1, b - 1) for a, b in segs]) if segs else np.array([], int)
    if len(idx) < 10:
        return dict(ok=False, reason="too few contiguous DVL samples")
    t = log.dvl_t[idx]
    dvdt = (log.dvl_v[idx + 1] - log.dvl_v[idx - 1]) / (log.dvl_t[idx + 1] - log.dvl_t[idx - 1])[:, None]

    def pair(lag):
        ts = log.imu_t - lag                                  # undo a suspected IMU clock lead
        ok = (t >= ts[0]) & (t <= ts[-1])
        w = _interp(ts, log.imu_gyro, t[ok])
        # ponytail: w x v uses gyro in IMU frame; exact only for small mount angles or low rates
        a_dvl = dvdt[ok] + np.cross(w, log.dvl_v[idx][ok])
        return a_dvl, _interp(ts, a_imu, t[ok])

    a_dvl0, a_imu0 = pair(0.0)
    if len(a_dvl0) < 10:
        return dict(ok=False, reason="IMU and DVL time ranges do not overlap enough")
    excite = _rms(a_dvl0)
    if excite < TH["excite_mps2"]:
        return dict(ok=False, reason="insufficient acceleration excitation", excitation=round(excite, 3))
    gdir = log.imu_acc.mean(0) / g
    zhat = np.array([0.0, 0.0, 1.0])
    # Gravity anchor: one precise direction pair, weighted as half the evidence (g^2 per sample).
    anchor = (zhat, gdir, 0.5 * len(a_dvl0) * g ** 2)

    def resid(a_i, a_d, R):
        """Absolute m/s^2: accel residual plus gravity-anchor residual (healthy noise floor ~0.09)."""
        acc = np.mean(np.sum((a_i - a_d @ R.T) ** 2, -1))
        anc = 0.5 * g ** 2 * np.sum((gdir - R @ zhat) ** 2)
        return float(np.sqrt(acc + anc))

    R0, ang0, _ = _kabsch(a_dvl0, a_imu0, anchor)
    r00 = resid(a_imu0, a_dvl0, np.eye(3))
    rR0 = resid(a_imu0, a_dvl0, R0)
    best = None
    for lag in LAG_GRID:
        a_d, a_i = pair(float(lag))
        if len(a_d) < 10:
            continue
        r_lag_only = resid(a_i, a_d, np.eye(3))
        Rl, ang, axis = _kabsch(a_d, a_i, anchor)
        r_both = resid(a_i, a_d, Rl)
        if best is None or r_both < best["unexplained"]:
            best = dict(lag_s=round(float(lag), 3), rot_deg=round(ang, 2), rot_axis=axis,
                        lag_only=r_lag_only, unexplained=r_both)
    if best is None:
        return dict(ok=False, reason="no lag in the grid leaves enough overlapping samples")
    rm = lambda a, b: float(np.sqrt(max(a * a - b * b, 0.0)))   # residual removed, m/s^2; no division
    return dict(ok=True, excitation=round(excite, 3), raw=round(r00, 3),
                rot_removed=round(rm(r00, rR0), 3), rot_only_deg=round(ang0, 2),
                lag_removed=round(rm(r00, best["lag_only"]), 3),
                unexplained=round(best["unexplained"], 3), lag_s=best["lag_s"],
                rot_deg=best["rot_deg"], rot_axis=best["rot_axis"])


def depth_dvl_check(log: SensorLog) -> dict:
    """RMS of (world-z DVL velocity via IMU orientation + depth rate), m/s. Healthy ~0.04."""
    R = _rot_batch(_interp(log.imu_t, log.imu_q, log.dvl_t))
    vz_world = np.einsum("nij,nj->ni", R, log.dvl_v)[:, 2]
    segs, _ = _segments(log.depth_t)
    idx = np.concatenate([np.arange(a + 1, b - 1) for a, b in segs]) if segs else np.array([], int)
    if len(idx) < 3:
        return dict(vertical_excitation=0.0, mismatch_mps=0.0, reason="too few contiguous depth samples")
    ddepth = (log.depth[idx + 1] - log.depth[idx - 1]) / (log.depth_t[idx + 1] - log.depth_t[idx - 1])
    vz = np.interp(log.depth_t[idx], log.dvl_t, vz_world)
    return dict(vertical_excitation=round(_rms(ddepth[:, None]), 3),
                mismatch_mps=round(_rms((vz + ddepth)[:, None]), 3))


# ---- ranking -----------------------------------------------------------------------
def _clip(x):
    return float(np.clip(x, 0.0, 1.0))


def _ranked(score):
    return sorted(score.items(), key=lambda kv: -kv[1])


def diagnose(logs: list[SensorLog] | SensorLog) -> Diagnosis:
    logs = [logs] if isinstance(logs, SensorLog) else list(logs)
    if not logs:
        raise InvalidSensorLog("diagnose() needs at least one SensorLog")
    logs = [validate(l) for l in logs]          # drops non-finite rows; raises on unusable input
    per = [dict(gravity=gravity_axis_check(l), gaps=dvl_gap_check(l),
                imu_dvl=imu_dvl_check(l), depth=depth_dvl_check(l)) for l in logs]
    gaps = max((c["gaps"] for c in per), key=lambda g: g["longest_gap_s"])
    cross = max((c["imu_dvl"] for c in per), key=lambda c: c.get("excitation", -1))
    depth = max((c["depth"] for c in per), key=lambda c: c["mismatch_mps"])
    checks = dict(gravity=per[0]["gravity"], gaps=gaps, imu_dvl=cross, depth=depth, n_logs=len(logs))
    notes = []
    score = {h: 0.0 for h in HYPOTHESES}
    tests_done = len(logs)

    if gaps["longest_gap_s"] > TH["gap_s"]:
        score["dvl_dropout"] = _clip(0.5 + 0.5 * gaps["longest_gap_s"] / TH["gap_s"])

    if not cross["ok"]:
        notes.append("imu_dvl cross-check inconclusive: " + cross["reason"])
        if score["dvl_dropout"] > 0.5:
            return Diagnosis(_ranked(score), checks, None, notes)
        nxt = "surge_pulses" if tests_done == 1 else None
        return Diagnosis([(ABSTAIN, 1.0)] + _ranked(score), checks, nxt, notes)

    if abs(cross["lag_s"]) > TH["lag_s"]:
        score["timestamp_offset"] = _clip(cross["lag_removed"] / TH["evidence_mps2"]) \
            * _clip(abs(cross["lag_s"]) / TH["lag_s"] - 0.5)
    if cross["rot_deg"] > TH["rot_deg"]:
        s = _clip(cross["rot_removed"] / TH["evidence_mps2"]) * _clip(cross["rot_deg"] / TH["rot_deg"] - 0.5)
        if cross["rot_axis"] in "xy" and depth["mismatch_mps"] > TH["depth_mps"]:
            s = _clip(s + 0.2)     # independent confirmation from the depth cross-check
        score["mount_rotation_" + cross["rot_axis"]] = s
    if cross["unexplained"] > TH["unexplained_mps2"]:
        notes.append(f"best lag+rotation fit leaves {cross['unexplained']:.2f} m/s^2 unexplained "
                     f"(healthy floor ~0.09): outside the hypothesis set")
        return Diagnosis([(ABSTAIN, 1.0)] + _ranked(score), checks, None, notes)

    worst = max(v for h, v in score.items() if h != "healthy")
    score["healthy"] = _clip(1.0 - worst)
    rank = _ranked(score)
    next_test = None
    weak = cross["excitation"] < TH["excite_strong"]
    close = rank[0][1] - rank[1][1] < TH["close"]
    if tests_done == 1 and (weak or close) and score["dvl_dropout"] < 0.5:
        # next-test rule table (hand-set): low excitation -> surge; mount-vs-offset tie -> depth bob
        top2 = {family(rank[0][0]), family(rank[1][0])}
        next_test = "depth_bob" if (not weak and top2 == {"mount_rotation", "timestamp_offset"}) else "surge_pulses"
        notes.append(f"{'weak excitation' if weak else 'top-2 close'}: requesting {next_test}")
    return Diagnosis(rank, checks, next_test, notes)


def simple_checks(log: SensorLog) -> str:
    """Baseline with the existing FrameForge rules only (no cross-check, no offset test)."""
    log = validate(log)
    if gravity_axis_check(log)["fail"]:
        return "mount_rotation"
    if dvl_gap_check(log)["n_gaps"] > 0:
        return "dvl_dropout"
    return "healthy"
