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

import re
from dataclasses import dataclass, field

import numpy as np

from .types import InvalidSensorLog, SensorLog, validate  # neutral module: no sim/evaluator imports

HYPOTHESES = ["healthy", "mount_rotation_x", "mount_rotation_y", "mount_rotation_z",
              "timestamp_offset", "dvl_dropout"]
ABSTAIN = "abstain"
TH = dict(gap_s=0.5, lag_s=0.04, rot_deg=5.0, excite_mps2=0.12, excite_strong=0.5,
          evidence_mps2=0.3, unexplained_mps2=0.2, depth_mps=0.1, close=0.2,
          drift=0.002, depth_excite_mps=0.05, gyro_bias_rps=0.02)
LAG_GRID = np.arange(-1.0, 1.0 + 1e-9, 0.01)
# IMU clock rate error, searched only when a constant lag cannot close the residual
DRIFT_GRID = [d * sgn for d in (0.0025, 0.005, 0.01, 0.02, 0.04) for sgn in (1, -1)]


def family(h: str) -> str:
    """Hypothesis -> fault family: drops a trailing axis (_x/_y/_z) or joint (_j1, _j2, ...)."""
    return re.sub(r"_(?:[xyz]|j\d+)$", "", h)


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


def gyro_orientation_check(log: SensorLog) -> dict:
    """Mean (gyro - rate implied by consecutive imu_q), rad/s, in the IMU frame.  A mount
    rotates both streams together, a clock fault shifts both together, so only a gyro bias
    (or an orientation filter that ignores the gyro) leaves a persistent difference.
    Healthy floor ~1e-3 (0.005 rad/s white noise averaged over 2000 samples)."""
    q = log.imu_q
    w0, v0 = q[:-1, :1], q[:-1, 1:]
    w1, v1 = q[1:, :1], q[1:, 1:]
    # dq = conj(q_k) * q_{k+1}  (rotation of the IMU frame between samples)
    dw = w0 * w1 + np.sum(v0 * v1, -1, keepdims=True)
    dv = w0 * v1 - w1 * v0 - np.cross(v0, v1)
    n = np.linalg.norm(dv, axis=-1, keepdims=True)
    ang = 2.0 * np.arctan2(n, np.abs(dw)) * np.sign(dw)      # shortest arc, radians
    dt = np.diff(log.imu_t)[:, None]
    rate_q = np.where(n > 1e-12, dv / np.maximum(n, 1e-12) * ang, 0.0) / dt
    rate_g = 0.5 * (log.imu_gyro[:-1] + log.imu_gyro[1:])
    bias = (rate_g - rate_q).mean(0)
    return dict(bias_rps=[round(float(b), 4) for b in bias],
                bias_norm_rps=round(float(np.linalg.norm(bias)), 4))


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

    t_mid = 0.5 * (t[0] + t[-1])

    def pair(lag, drift=0.0):
        # undo a suspected IMU clock lead, constant (lag) plus linear in time (drift about t_mid)
        ts = log.imu_t - lag - drift * (log.imu_t - t_mid)
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

    def resid(a_i, a_d, R, anchored=True):
        """Absolute m/s^2: accel residual plus gravity-anchor residual (healthy noise floor ~0.09)."""
        acc = np.mean(np.sum((a_i - a_d @ R.T) ** 2, -1))
        anc = 0.5 * g ** 2 * np.sum((gdir - R @ zhat) ** 2) if anchored else 0.0
        return float(np.sqrt(acc + anc))

    R0, ang0, _ = _kabsch(a_dvl0, a_imu0, anchor)
    r00 = resid(a_imu0, a_dvl0, np.eye(3))
    rR0 = resid(a_imu0, a_dvl0, R0)
    def search(lags, drift, best=None):
        for lag in lags:
            a_d, a_i = pair(float(lag), drift)
            if len(a_d) < 10:
                continue
            r_lag_only = resid(a_i, a_d, np.eye(3))
            Rl, ang, axis = _kabsch(a_d, a_i, anchor)
            r_both = resid(a_i, a_d, Rl)
            if best is None or r_both < best["unexplained"]:
                best = dict(lag_s=round(float(lag), 3), drift=drift, rot_deg=round(ang, 2),
                            rot_axis=axis, lag_only=r_lag_only, unexplained=r_both)
        return best

    best = search(LAG_GRID, 0.0)
    if best is None:
        return dict(ok=False, reason="no lag in the grid leaves enough overlapping samples")
    if best["unexplained"] > TH["unexplained_mps2"]:
        # ponytail: coarse (50 ms) lag grid per drift value, then refine lag at the best drift
        for d in DRIFT_GRID:
            best = search(LAG_GRID[::5], d, best)
        if best["drift"]:
            best = search(np.arange(best["lag_s"] - 0.05, best["lag_s"] + 0.05 + 1e-9, 0.01), best["drift"], best)
    prior_violated = False
    if best["unexplained"] > TH["unexplained_mps2"]:
        # The level-at-rest prior (gravity anchor) may itself be wrong: a vehicle that trims at an
        # angle. Refit at the same lag/drift without the anchor; accept only if that closes the fit.
        # ponytail: unanchored Kabsch is degenerate under single-axis excitation (rotation about
        # the excited axis is unobservable); the depth cross-check and a depth_bob request cover x/y.
        a_d, a_i = pair(best["lag_s"], best["drift"])
        Ru, angu, axu = _kabsch(a_d, a_i, None)
        ru = resid(a_i, a_d, Ru, anchored=False)
        if ru <= TH["unexplained_mps2"]:
            prior_violated = True
            best = dict(best, rot_deg=round(angu, 2), rot_axis=axu, unexplained=ru,
                        lag_only=resid(a_i, a_d, np.eye(3), anchored=False))
            r00, rR0 = resid(a_imu0, a_dvl0, np.eye(3), False), resid(a_imu0, a_dvl0, _kabsch(a_dvl0, a_imu0)[0], False)
    rm = lambda a, b: float(np.sqrt(max(a * a - b * b, 0.0)))   # residual removed, m/s^2; no division
    return dict(ok=True, excitation=round(excite, 3), raw=round(r00, 3), prior_violated=prior_violated,
                tilt_deg=round(float(np.degrees(np.arccos(np.clip(gdir @ zhat, -1, 1)))), 2),
                rot_removed=round(rm(r00, rR0), 3), rot_only_deg=round(ang0, 2),
                lag_removed=round(rm(r00, best["lag_only"]), 3),
                unexplained=round(best["unexplained"], 3), lag_s=best["lag_s"],
                drift=best["drift"], lag_span_s=round(abs(best["drift"]) * (t[-1] - t[0]) / 2, 3),
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


def depth_sensitivity(log: SensorLog, axis: str, deg: float) -> float:
    """Predicted depth-check mismatch (m/s) if a mount of (axis, deg) were real, i.e. how much the
    DVL vertical velocity through imu_q would change.  Below TH['depth_mps'] the depth check
    cannot see that mount with this motion (e.g. roll needs sway)."""
    a = np.radians(deg) / 2.0
    v = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[axis]
    Rm = _rot_batch(np.array([[np.cos(a), *(np.sin(a) * np.asarray(v, float))]]))[0]
    R = _rot_batch(_interp(log.imu_t, log.imu_q, log.dvl_t))
    vz = np.einsum("nij,nj->ni", R, log.dvl_v)[:, 2]
    vz2 = np.einsum("nij,nj->ni", R, log.dvl_v @ Rm)[:, 2]
    return round(_rms((vz - vz2)[:, None]), 3)


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
    per = [dict(gravity=gravity_axis_check(l), gaps=dvl_gap_check(l), gyro=gyro_orientation_check(l),
                imu_dvl=imu_dvl_check(l), depth=depth_dvl_check(l)) for l in logs]
    gyro = max((c["gyro"] for c in per), key=lambda c: c["bias_norm_rps"])
    gaps = max((c["gaps"] for c in per), key=lambda g: g["longest_gap_s"])
    cross = max((c["imu_dvl"] for c in per), key=lambda c: c.get("excitation", -1))
    i_depth = max(range(len(per)), key=lambda i: per[i]["depth"]["vertical_excitation"])
    depth = per[i_depth]["depth"]
    checks = dict(gravity=per[0]["gravity"], gaps=gaps, gyro=gyro, imu_dvl=cross, depth=depth, n_logs=len(logs))
    notes = []
    score = {h: 0.0 for h in HYPOTHESES}
    tests_done = len(logs)

    if gaps["longest_gap_s"] > TH["gap_s"]:
        score["dvl_dropout"] = _clip(0.5 + 0.5 * gaps["longest_gap_s"] / TH["gap_s"])
    if gyro["bias_norm_rps"] > TH["gyro_bias_rps"]:
        notes.append(f"gyro disagrees with imu_q by a steady {gyro['bias_norm_rps']:.3f} rad/s "
                     f"(axes {gyro['bias_rps']}): gyro bias, outside the hypothesis set")
        return Diagnosis([(ABSTAIN, 1.0)] + _ranked(score), checks, None, notes)

    if not cross["ok"]:
        notes.append("imu_dvl cross-check inconclusive: " + cross["reason"])
        if score["dvl_dropout"] > 0.5:
            return Diagnosis(_ranked(score), checks, None, notes)
        nxt = "surge_pulses" if tests_done == 1 else None
        return Diagnosis([(ABSTAIN, 1.0)] + _ranked(score), checks, nxt, notes)

    if cross["prior_violated"]:
        notes.append(f"mean specific force is {cross['tilt_deg']:.1f} deg off body +Z and the gravity-anchored "
                     "fit does not close: level-at-rest prior dropped (vehicle trim or a mount, see depth check)")
    lag_eff = abs(cross["lag_s"]) + cross["lag_span_s"]       # constant lag plus drift excursion
    if lag_eff > TH["lag_s"] or abs(cross["drift"]) >= TH["drift"]:
        score["timestamp_offset"] = _clip(cross["lag_removed"] / TH["evidence_mps2"]) \
            * _clip(lag_eff / TH["lag_s"] - 0.5)
        if cross["drift"]:
            notes.append(f"IMU clock rate error {cross['drift']:+.2%} fits better than a constant lag")
    need_depth = False
    if cross["rot_deg"] > TH["rot_deg"]:
        s = _clip(cross["rot_removed"] / TH["evidence_mps2"]) * _clip(cross["rot_deg"] / TH["rot_deg"] - 0.5)
        if cross["rot_axis"] in "xy":
            # A roll/pitch mount corrupts imu_q, so DVL-through-attitude disagrees with depth rate.
            # A trimmed vehicle does not: imu_q is right, only the level-at-rest prior is wrong.
            sens = depth_sensitivity(logs[i_depth], cross["rot_axis"], cross["rot_deg"])
            checks["depth"] = dict(depth, predicted_if_mount_mps=sens)
            if depth["vertical_excitation"] < TH["depth_excite_mps"]:
                need_depth = True                                  # cannot tell yet
            elif sens < TH["depth_mps"]:
                notes.append(f"depth cross-check cannot see a {cross['rot_axis']} rotation with this "
                             "motion (needs sway for roll): mount vs trim unresolved, mount kept")
            elif depth["mismatch_mps"] > TH["depth_mps"]:
                s = _clip(s + 0.2)                                 # confirmed by depth
            else:
                s *= 0.3
                notes.append("depth cross-check agrees with imu_q: apparent "
                             f"{cross['rot_axis']} rotation is vehicle trim, not a mount")
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
    if tests_done == 1 and need_depth and score["mount_rotation_" + cross["rot_axis"]] > 0.3:
        next_test = "depth_bob"
        notes.append("roll/pitch rotation unconfirmed without vertical motion: requesting depth_bob")
    elif tests_done == 1 and (weak or close) and score["dvl_dropout"] < 0.5:
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
