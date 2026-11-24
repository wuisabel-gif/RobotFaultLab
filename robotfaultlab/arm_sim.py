"""Arm scenario generation (evaluator side).  Joint trajectories are prescribed analytically
and torques come from inverse dynamics of the TRUE arm (nominal, or modified for unknown
faults), so the truth is exact.  generate() returns (ArmLog, ArmSecret); only ArmLog goes to
the diagnostic.

This simulator is new, deliberately small code (no upstream arm simulator is used): a planar
2R arm with viscous friction and white sensor noise.  No flexibility, backlash, Coulomb
friction, or controller.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .arm_types import NOMINAL, ArmLog, inverse_dynamics

ENC_HZ, TAU_HZ, DURATION_S = 100.0, 50.0, 12.0
NOISE = dict(enc=1e-4, tau=0.05)     # rad, N m
L2 = 0.30                            # link-2 length (payload attaches at the tip), m


@dataclass
class ArmSecret:
    family: str              # healthy | encoder_offset | timestamp_offset | torque_dropout | unknown
    profile: str
    seed: int
    params: dict = field(default_factory=dict)


# ---- motion profiles: list of (centre, [(amplitude, omega, phase), ...]) per joint ----
def _sines(seed):
    r = np.random.default_rng([seed, 99])
    return [(r.uniform(-0.3, 0.6), [(r.uniform(0.15, 0.35), r.uniform(0.5, 2.5), r.uniform(0, 6.28)) for _ in range(3)])
            for _ in range(2)]


PROFILES = {
    "hold":          lambda seed: [(0.6, []), (-0.8, [])],
    "sweep":         lambda seed: [(0.5, [(0.6, 0.8, 0.0)]), (-0.7, [(0.8, 1.3, 0.5)])],
    "shoulder_only": lambda seed: [(0.3, [(0.7, 1.0, 0.0)]), (-0.5, [])],
    "elbow_only":    lambda seed: [(0.4, []), (-0.6, [(0.9, 1.2, 0.0)])],
    "fast_wave":     lambda seed: [(0.2, [(0.4, 2.5, 0.0)]), (0.3, [(0.5, 3.1, 1.0)])],
    "random_smooth": _sines,
}


def _trajectory(profile, seed, t):
    q, qd, qdd = (np.zeros((len(t), 2)) for _ in range(3))
    for j, (centre, terms) in enumerate(PROFILES[profile](seed)):
        q[:, j] = centre
        for a, w, ph in terms:
            q[:, j] += a * np.sin(w * t + ph)
            qd[:, j] += a * w * np.cos(w * t + ph)
            qdd[:, j] -= a * w * w * np.sin(w * t + ph)
    return q, qd, qdd


def generate(secret: ArmSecret) -> tuple[ArmLog, ArmSecret]:
    rng = np.random.default_rng(secret.seed)
    p, fam = secret.params, secret.family
    enc_t = np.arange(0.0, DURATION_S, 1.0 / ENC_HZ)
    tau_t = np.arange(0.0, DURATION_S, 1.0 / TAU_HZ)
    model = NOMINAL
    if fam == "unknown" and p["kind"] == "payload":          # point mass gripped at the tip
        mp, m2 = p["mass_kg"], NOMINAL.m2
        lc2 = (m2 * NOMINAL.lc2 + mp * L2) / (m2 + mp)
        I2 = NOMINAL.I2 + m2 * (lc2 - NOMINAL.lc2) ** 2 + mp * (L2 - lc2) ** 2
        model = replace(NOMINAL, m2=m2 + mp, lc2=lc2, I2=I2)
    q = _trajectory(secret.profile, secret.seed, enc_t)[0]
    tau = inverse_dynamics(model, *_trajectory(secret.profile, secret.seed, tau_t))
    log = ArmLog(enc_t, q + rng.normal(0, NOISE["enc"], q.shape),
                 tau_t, tau + rng.normal(0, NOISE["tau"], tau.shape))
    if fam == "encoder_offset":
        log.enc_q[:, p["joint"] - 1] += p["offset_rad"]     # encoder zero is wrong on one joint
    elif fam == "timestamp_offset":
        log.enc_t = log.enc_t + p["offset_s"]               # encoder clock ahead of torque clock
    elif fam == "torque_dropout":
        keep = ~((tau_t >= p["start_s"]) & (tau_t < p["start_s"] + p["duration_s"]))
        log.tau_t, log.tau = log.tau_t[keep], log.tau[keep]
    elif fam == "unknown":
        if p["kind"] == "torque_scale":
            log.tau[:, p["joint"] - 1] *= p["scale"]
        elif p["kind"] != "payload":
            raise ValueError(p["kind"])
    elif fam != "healthy":
        raise ValueError(fam)
    return log, secret
