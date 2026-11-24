"""Sensor-only arm diagnostic.  Input is ArmLog (encoders + joint torques) and the nominal
ArmModel; nothing from any simulator or the evaluator is imported.

Checks:
  torque_gaps    torque-stream stamp gaps vs nominal period                 (simple/existing)
  model_torque   cross-check: torque predicted from the encoders through the nominal inverse
                 dynamics vs measured torque, fitted jointly for a time lag and a constant
                 encoder offset per joint.  An encoder zero error shows as a constant angle
                 offset; a clock offset shows as a lag; anything else (payload, sensor gain)
                 leaves torque unexplained and the diagnostic abstains.

Thresholds in TH were set by hand on evaluate.arm_dev_cases() only.
"""
from __future__ import annotations

import numpy as np

from .arm_types import NOMINAL, ArmLog, InvalidSensorLog, inverse_dynamics, validate
from .diagnose import ABSTAIN, Diagnosis, _clip, _interp, _ranked, _rms

HYPOTHESES = ["healthy", "encoder_offset_j1", "encoder_offset_j2", "timestamp_offset", "torque_dropout"]
TH = dict(gap_s=0.2, lag_s=0.015, off_rad=0.015, excite_nm=0.3, evidence_nm=0.15,
          unexplained_nm=0.12, close=0.2)
LAG_GRID = np.arange(-0.5, 0.5 + 1e-9, 0.005)
SMOOTH_S, EDGE_S = 0.15, 0.6


def torque_gap_check(log: ArmLog) -> dict:
    dts = np.diff(log.tau_t)
    nominal = float(np.median(dts))
    gaps = dts[dts > 3 * nominal]
    return dict(nominal_dt=round(nominal, 4), n_gaps=int(len(gaps)),
                longest_gap_s=round(float(gaps.max()) if len(gaps) else 0.0, 3))


def _kinematics(log: ArmLog):
    """Smoothed q and its first two derivatives on the encoder clock, edges trimmed.
    ponytail: double box filter + np.gradient; fine below ~1 Hz joint motion, a
    Savitzky-Golay or spline fit is the upgrade for faster arms."""
    dt = float(np.median(np.diff(log.enc_t)))
    k = max(3, int(round(SMOOTH_S / dt)) | 1)
    box = np.ones(k) / k
    q = np.stack([np.convolve(np.convolve(log.enc_q[:, j], box, "same"), box, "same") for j in range(2)], -1)
    qd = np.gradient(q, log.enc_t, axis=0)
    qdd = np.gradient(qd, log.enc_t, axis=0)
    e = int(round(EDGE_S / dt))
    return log.enc_t[e:-e], q[e:-e], qd[e:-e], qdd[e:-e]


def model_torque_check(log: ArmLog, model=NOMINAL) -> dict:
    t, q, qd, qdd = _kinematics(log)
    if len(t) < 50:
        return dict(ok=False, reason="too few encoder samples after smoothing")
    pred0 = inverse_dynamics(model, q, qd, qdd)
    excite = _rms(pred0 - pred0.mean(0))
    if excite < TH["excite_nm"]:
        # A static pose gives two torques for two unknown offsets: any offset, payload or gain
        # error fits exactly, and a clock offset is invisible.
        return dict(ok=False, reason="insufficient torque variation (arm nearly static)",
                    excitation=round(excite, 3))

    def fit(lag, with_offset):
        """Residual (N m rms) and offset after undoing a suspected encoder-clock lead."""
        ts = t - lag
        ok = (log.tau_t >= ts[0]) & (log.tau_t <= ts[-1])
        if ok.sum() < 20:
            return None
        meas, tt = log.tau[ok], log.tau_t[ok]
        d = np.zeros(2)
        pred = _interp(ts, pred0, tt)
        if with_offset:
            for _ in range(4):                      # Gauss-Newton on the two joint offsets
                base = inverse_dynamics(model, q - d, qd, qdd)
                J = np.stack([(inverse_dynamics(model, q - d - 1e-4 * np.eye(2)[j], qd, qdd) - base) / -1e-4
                              for j in range(2)], -1)             # d tau / d q, (N,2,2)
                pred = _interp(ts, base, tt)
                Ji = np.stack([_interp(ts, J[:, :, j], tt) for j in range(2)], -1)
                step, *_ = np.linalg.lstsq(Ji.reshape(-1, 2), (meas - pred).reshape(-1), rcond=None)
                d = d - step                         # pred uses q - d, so d is the encoder offset
            pred = _interp(ts, inverse_dynamics(model, q - d, qd, qdd), tt)
        return _rms(meas - pred), d

    r00 = fit(0.0, False)
    if r00 is None:
        return dict(ok=False, reason="encoder and torque time ranges do not overlap enough")
    r_off0 = fit(0.0, True)
    best = lag_only = None
    for lag in LAG_GRID:
        a, b = fit(float(lag), False), fit(float(lag), True)
        if a is None:
            continue
        if lag_only is None or a[0] < lag_only:
            lag_only = a[0]
        if best is None or b[0] < best[0]:
            best = (b[0], float(lag), b[1])
    rm = lambda a, b: float(np.sqrt(max(a * a - b * b, 0.0)))    # residual removed, N m; no division
    return dict(ok=True, excitation=round(excite, 3), raw=round(r00[0], 3),
                off_removed=round(rm(r00[0], r_off0[0]), 3), lag_removed=round(rm(r00[0], lag_only), 3),
                unexplained=round(best[0], 3), lag_s=round(best[1], 3),
                offset_rad=[round(float(x), 4) for x in best[2]])


def diagnose(logs: list[ArmLog] | ArmLog) -> Diagnosis:
    logs = [logs] if isinstance(logs, ArmLog) else list(logs)
    if not logs:
        raise InvalidSensorLog("diagnose() needs at least one ArmLog")
    logs = [validate(l) for l in logs]
    per = [dict(gaps=torque_gap_check(l), model=model_torque_check(l)) for l in logs]
    gaps = max((c["gaps"] for c in per), key=lambda g: g["longest_gap_s"])
    cross = max((c["model"] for c in per), key=lambda c: c.get("excitation", -1))
    checks = dict(gaps=gaps, model_torque=cross, n_logs=len(logs))
    notes, score, first = [], {h: 0.0 for h in HYPOTHESES}, len(logs) == 1

    if gaps["longest_gap_s"] > TH["gap_s"]:
        score["torque_dropout"] = _clip(0.5 + 0.5 * gaps["longest_gap_s"] / TH["gap_s"])
    if not cross["ok"]:
        notes.append("model_torque cross-check inconclusive: " + cross["reason"])
        if score["torque_dropout"] > 0.5:
            return Diagnosis(_ranked(score), checks, None, notes)
        return Diagnosis([(ABSTAIN, 1.0)] + _ranked(score), checks, "sweep" if first else None, notes)

    if abs(cross["lag_s"]) > TH["lag_s"]:
        score["timestamp_offset"] = _clip(cross["lag_removed"] / TH["evidence_nm"]) \
            * _clip(abs(cross["lag_s"]) / TH["lag_s"] - 0.5)
    for j, off in enumerate(cross["offset_rad"]):
        if abs(off) > TH["off_rad"]:
            score[f"encoder_offset_j{j + 1}"] = _clip(cross["off_removed"] / TH["evidence_nm"]) \
                * _clip(abs(off) / TH["off_rad"] - 0.5)
    if cross["unexplained"] > TH["unexplained_nm"]:
        notes.append(f"best lag+offset fit leaves {cross['unexplained']:.2f} N m unexplained: "
                     "outside the hypothesis set (payload, torque-sensor gain, friction...)")
        return Diagnosis([(ABSTAIN, 1.0)] + _ranked(score), checks, None, notes)

    score["healthy"] = _clip(1.0 - max(v for h, v in score.items() if h != "healthy"))
    rank = _ranked(score)
    next_test = None
    if first and rank[0][1] - rank[1][1] < TH["close"] and score["torque_dropout"] < 0.5:
        next_test = "elbow_only"      # moves one joint alone: separates joint-1 from joint-2 offsets
        notes.append("top-2 close: requesting elbow_only")
    return Diagnosis(rank, checks, next_test, notes)


def simple_checks(log: ArmLog) -> str:
    """Baseline: the usual single-threshold model-residual monitor plus a gap rule.  It flags
    that torque disagrees with the model but cannot say why, never fits a lag, never abstains."""
    log = validate(log)
    if torque_gap_check(log)["longest_gap_s"] > TH["gap_s"]:
        return "torque_dropout"
    t, q, qd, qdd = _kinematics(log)
    ok = (log.tau_t >= t[0]) & (log.tau_t <= t[-1])
    resid = _rms(log.tau[ok] - _interp(t, inverse_dynamics(NOMINAL, q, qd, qdd), log.tau_t[ok]))
    return "encoder_offset" if resid > 3 * 0.05 * np.sqrt(2) else "healthy"
