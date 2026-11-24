"""Neutral sensor-only data type.  This module imports nothing from the simulator
or evaluator; diagnose.py and frameforge.py depend on it alone.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np


class InvalidSensorLog(ValueError):
    """Raised by validate() for inputs the diagnostic cannot reason about."""


@dataclass
class SensorLog:
    """Sensor-only data. No labels, no truth poses."""
    imu_t: np.ndarray      # (N,)
    imu_acc: np.ndarray    # (N,3) specific force, IMU frame
    imu_gyro: np.ndarray   # (N,3) body rate, IMU frame
    imu_q: np.ndarray      # (N,4) wxyz, world<-IMU
    dvl_t: np.ndarray      # (M,)
    dvl_v: np.ndarray      # (M,3) body-frame velocity
    depth_t: np.ndarray    # (K,)
    depth: np.ndarray      # (K,) metres below surface (positive down)


MIN_IMU, MIN_DVL, MIN_DEPTH = 50, 10, 5
_SHAPES = dict(imu_t=1, imu_acc=3, imu_gyro=3, imu_q=4, dvl_t=1, dvl_v=3, depth_t=1, depth=1)


def validate_streams(log, cls, shapes, groups) -> dict:
    """Generic stream validation for any log dataclass.  shapes: field -> 1 (1-D) or column
    count; groups: (time_field, value_fields, min_samples).  Returns cleaned float arrays with
    non-finite rows dropped; raises InvalidSensorLog on anything unusable."""
    if not isinstance(log, cls):
        raise InvalidSensorLog(f"expected {cls.__name__}, got {type(log).__name__}")
    arr = {}
    for f in fields(cls):
        try:
            a = np.asarray(getattr(log, f.name), dtype=float)
        except (TypeError, ValueError) as e:
            raise InvalidSensorLog(f"{f.name}: not numeric ({e})") from None
        want = shapes[f.name]
        if want == 1:
            if a.ndim != 1:
                raise InvalidSensorLog(f"{f.name}: expected 1-D, got shape {a.shape}")
        elif a.ndim != 2 or a.shape[1] != want:
            raise InvalidSensorLog(f"{f.name}: expected (N,{want}), got shape {a.shape}")
        arr[f.name] = a
    for t_key, keys, n_min in groups:
        n = len(arr[t_key])
        if any(len(arr[k]) != n for k in keys):
            raise InvalidSensorLog(f"{t_key}: row count mismatch")
        ok = np.isfinite(arr[t_key])
        for k in keys:
            ok &= np.all(np.isfinite(arr[k]), axis=-1) if arr[k].ndim == 2 else np.isfinite(arr[k])
        if ok.sum() < n_min:
            raise InvalidSensorLog(f"{t_key}: only {int(ok.sum())} finite samples, need {n_min}")
        for k in (t_key, *keys):
            arr[k] = arr[k][ok]
        if np.any(np.diff(arr[t_key]) <= 0):
            raise InvalidSensorLog(f"{t_key}: time stamps must be strictly increasing")
    return arr


def validate(log: SensorLog) -> SensorLog:
    """Return a copy with non-finite samples dropped; raise InvalidSensorLog if what is
    left is too short, mis-shaped, or has non-increasing time stamps."""
    arr = validate_streams(log, SensorLog, _SHAPES, [
        ("imu_t", ("imu_acc", "imu_gyro", "imu_q"), MIN_IMU),
        ("dvl_t", ("dvl_v",), MIN_DVL),
        ("depth_t", ("depth",), MIN_DEPTH)])
    qn = np.linalg.norm(arr["imu_q"], axis=1)
    if np.any(qn < 1e-6):
        raise InvalidSensorLog("imu_q: zero-norm quaternion")
    arr["imu_q"] = arr["imu_q"] / qn[:, None]
    return SensorLog(**arr)
