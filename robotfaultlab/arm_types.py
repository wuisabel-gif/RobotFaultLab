"""Neutral arm data types: sensor-only log plus the nominal (datasheet/URDF) arm model.
Imports nothing from any simulator or the evaluator.

The arm is a planar two-link manipulator in a vertical plane: q1 is the shoulder angle from
horizontal, q2 the elbow angle relative to link 1, gravity along -y.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import InvalidSensorLog, validate_streams  # noqa: F401  (re-exported)


@dataclass
class ArmLog:
    """Sensor-only data. No labels, no true joint angles."""
    enc_t: np.ndarray    # (N,)   encoder clock
    enc_q: np.ndarray    # (N,2)  joint angles as reported by the encoders, rad
    tau_t: np.ndarray    # (M,)   torque-sensor clock
    tau: np.ndarray      # (M,2)  joint torques, N m


@dataclass(frozen=True)
class ArmModel:
    """Nominal parameters, public knowledge like a URDF (not evaluator secrets)."""
    l1: float = 0.40      # link-1 length, m
    lc1: float = 0.20     # link-1 centre of mass from shoulder, m
    lc2: float = 0.15     # link-2 centre of mass from elbow, m
    m1: float = 2.0       # kg
    m2: float = 1.0
    I1: float = 0.03      # kg m^2 about each link's centre of mass
    I2: float = 0.01
    b1: float = 0.3       # viscous friction, N m s/rad
    b2: float = 0.1
    g: float = 9.81


NOMINAL = ArmModel()
MIN_ENC, MIN_TAU = 100, 50


def inverse_dynamics(m: ArmModel, q, qd, qdd) -> np.ndarray:
    """Joint torques (N,2) for joint trajectories (N,2): M(q) qdd + C(q,qd) qd + g(q) + B qd."""
    q1, q2 = q[:, 0], q[:, 1]
    c2, s2 = np.cos(q2), np.sin(q2)
    M11 = m.I1 + m.I2 + m.m1 * m.lc1 ** 2 + m.m2 * (m.l1 ** 2 + m.lc2 ** 2 + 2 * m.l1 * m.lc2 * c2)
    M12 = m.I2 + m.m2 * (m.lc2 ** 2 + m.l1 * m.lc2 * c2)
    M22 = m.I2 + m.m2 * m.lc2 ** 2
    h = m.m2 * m.l1 * m.lc2 * s2
    g12 = m.m2 * m.g * m.lc2 * np.cos(q1 + q2)
    t1 = (M11 * qdd[:, 0] + M12 * qdd[:, 1] - h * (2 * qd[:, 0] * qd[:, 1] + qd[:, 1] ** 2)
          + (m.m1 * m.lc1 + m.m2 * m.l1) * m.g * np.cos(q1) + g12 + m.b1 * qd[:, 0])
    t2 = M12 * qdd[:, 0] + M22 * qdd[:, 1] + h * qd[:, 0] ** 2 + g12 + m.b2 * qd[:, 1]
    return np.stack([t1, t2], -1)


def validate(log: ArmLog) -> ArmLog:
    """Copy with non-finite rows dropped; InvalidSensorLog if too short, mis-shaped or
    non-monotonic."""
    arr = validate_streams(log, ArmLog, dict(enc_t=1, enc_q=2, tau_t=1, tau=2),
                           [("enc_t", ("enc_q",), MIN_ENC), ("tau_t", ("tau",), MIN_TAU)])
    return ArmLog(**arr)
