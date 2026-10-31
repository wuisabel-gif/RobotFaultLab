"""Scenario generation on top of AUVSimBench's physics core (Apache-2.0, imported
from sources/AUVSimBench unmodified; nothing is vendored).

Sensor sampling mirrors auv_sim_bench/sim_node.py (IMU: orientation, body rate,
specific force; DVL: body velocity; depth from world z) but without ROS, so the
same consistent-truth sensors can be produced in a plain script.

generate() returns (SensorLog, Secret).  SensorLog carries ONLY what a real
robot would log.  Secret carries the fault label/magnitude/profile and is for
the evaluator alone.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .types import SensorLog

_AUV = Path(__file__).resolve().parents[1] / "sources" / "AUVSimBench"
if str(_AUV) not in sys.path:
    sys.path.insert(0, str(_AUV))
from auv_sim_bench.dynamics import AUVDynamics, AUVParams, quat_mul, quat_to_rot  # noqa: E402

SIM_HZ, IMU_HZ, DVL_HZ, DEPTH_HZ = 200.0, 100.0, 8.0, 5.0
DURATION_S = 20.0
NOISE = dict(acc=0.05, gyro=0.005, dvl=0.01, depth=0.01)


@dataclass
class Secret:
    family: str              # healthy | mount_rotation | timestamp_offset | dvl_dropout | unknown
    profile: str
    seed: int
    params: dict = field(default_factory=dict)


# ---- motion profiles: t -> (force_body, torque_body) ---------------------------------
def _hover(t, rng):
    return np.zeros(3), np.zeros(3)


def _forward_cruise(t, rng):
    return (np.array([40.0, 0, 0]) if t > 1.0 else np.zeros(3)), np.zeros(3)


def _surge_pulses(t, rng):
    f = 40.0 if int(t // 2.0) % 2 == 0 else -25.0
    return np.array([f, 0, 0]), np.zeros(3)


def _yaw_sweep(t, rng):
    return np.array([25.0, 0, 0]), np.array([0, 0, 1.5 * np.sin(0.6 * t)])


def _depth_bob(t, rng):
    return np.array([15.0, 0, 30.0 * np.sin(0.8 * t)]), np.zeros(3)


def _trimmed_hover(t, rng):
    # constant pitch torque against the 0.02 m metacentric restoring moment (~2.9 N m max)
    # holds the vehicle ~18 deg nose-up: violates the diagnostic's level-at-rest prior
    return np.array([15.0, 0, 0]), np.array([0, 0.9, 0])


def _random_wrench(t, rng):
    # piecewise-constant random wrench, redrawn every second from the case rng
    k = int(t)
    r = np.random.default_rng([rng, k])
    return r.uniform([-30, -10, -25], [45, 10, 25]), r.uniform(-1.0, 1.0, 3) * np.array([0.3, 0.3, 1.0])


PROFILES = {
    "hover": _hover, "forward_cruise": _forward_cruise, "surge_pulses": _surge_pulses,
    "yaw_sweep": _yaw_sweep, "depth_bob": _depth_bob, "random_wrench": _random_wrench,
    "trimmed_hover": _trimmed_hover,
}


# ---- rotations ------------------------------------------------------------------------
def axis_angle_quat(axis: str, angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg) / 2.0
    v = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[axis]
    return np.array([np.cos(a), *(np.sin(a) * np.asarray(v, float))])


def _simulate(profile: str, seed: int):
    """Run AUVSimBench dynamics and sample ideal sensors at sim_node's rates."""
    rng = np.random.default_rng(seed)
    sim = AUVDynamics(AUVParams())
    wrench = PROFILES[profile]
    dt = 1.0 / SIM_HZ
    acc = {"imu": 0.0, "dvl": 0.0, "depth": 0.0}
    period = {"imu": 1.0 / IMU_HZ, "dvl": 1.0 / DVL_HZ, "depth": 1.0 / DEPTH_HZ}
    imu, dvl, dep = [], [], []
    t = 0.0
    for _ in range(int(DURATION_S * SIM_HZ)):
        f, tau = wrench(t, seed)
        s = sim.step(dt, f, tau)
        t += dt
        for k in acc:
            acc[k] += dt
        if acc["imu"] >= period["imu"] - 1e-9:
            acc["imu"] = 0.0
            imu.append((t, sim.specific_force.copy(), s.w.copy(), s.q.copy()))
        if acc["dvl"] >= period["dvl"] - 1e-9:
            acc["dvl"] = 0.0
            dvl.append((t, s.v.copy()))
        if acc["depth"] >= period["depth"] - 1e-9:
            acc["depth"] = 0.0
            dep.append((t, max(0.0, 0.0 - float(s.p[2]))))  # surface_z = 0 as in sim.yaml
    imu_t = np.array([r[0] for r in imu])
    imu_acc = np.array([r[1] for r in imu]) + rng.normal(0, NOISE["acc"], (len(imu), 3))
    imu_gyro = np.array([r[2] for r in imu]) + rng.normal(0, NOISE["gyro"], (len(imu), 3))
    imu_q = np.array([r[3] for r in imu])
    dvl_t = np.array([r[0] for r in dvl])
    dvl_v = np.array([r[1] for r in dvl]) + rng.normal(0, NOISE["dvl"], (len(dvl), 3))
    depth_t = np.array([r[0] for r in dep])
    depth = np.array([r[1] for r in dep]) + rng.normal(0, NOISE["depth"], len(dep))
    return SensorLog(imu_t, imu_acc, imu_gyro, imu_q, dvl_t, dvl_v, depth_t, depth)


def generate(secret: Secret) -> tuple[SensorLog, Secret]:
    """Apply the secret fault to ideal sensors. Only SensorLog goes to the diagnostic."""
    log = _simulate(secret.profile, secret.seed)
    p = secret.params
    fam = secret.family
    if fam == "mount_rotation":
        # IMU physically rotated by R_m w.r.t. body; the system assumes identity.
        qm = axis_angle_quat(p["axis"], p["angle_deg"])
        Rm = quat_to_rot(qm)
        log.imu_acc = log.imu_acc @ Rm          # (Rm^T f)^T = f^T Rm
        log.imu_gyro = log.imu_gyro @ Rm
        log.imu_q = np.array([quat_mul(q, qm) for q in log.imu_q])
    elif fam == "timestamp_offset":
        # fixed offset (IMU clock ahead of DVL/depth clock) and/or rate drift of the IMU clock
        log.imu_t = log.imu_t * (1.0 + p.get("drift", 0.0)) + p.get("offset_s", 0.0)
    elif fam == "dvl_dropout":
        win = (log.dvl_t >= p["start_s"]) & (log.dvl_t < p["start_s"] + p["duration_s"])
        if p.get("nan"):                         # driver keeps publishing, velocities NaN
            log.dvl_v = log.dvl_v.copy()
            log.dvl_v[win] = np.nan
        else:                                    # messages stop
            log.dvl_t, log.dvl_v = log.dvl_t[~win], log.dvl_v[~win]
    elif fam == "unknown":
        kind = p["kind"]
        if kind == "dvl_scale":
            log.dvl_v = log.dvl_v * p["scale"]
        elif kind == "accel_bias":
            log.imu_acc = log.imu_acc + np.array([p["bias_mps2"], 0, 0])
        elif kind == "gyro_bias":
            log.imu_gyro = log.imu_gyro + np.array([p["bias_rps"], 0, 0])
        elif kind == "dvl_beam_loss":            # sparse random NaN pings, no clean gap
            r = np.random.default_rng(secret.seed + 7)
            log.dvl_v = log.dvl_v.copy()
            log.dvl_v[r.random(len(log.dvl_t)) < p["frac"]] = np.nan
        else:
            raise ValueError(kind)
    elif fam != "healthy":
        raise ValueError(fam)
    return log, secret
