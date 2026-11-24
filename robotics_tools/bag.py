"""rosbag2 -> SensorLog.  Imports only the neutral types module.

Decoding is delegated to ros2bag_convert.py (Apache-2.0, stdlib-only, from
https://github.com/wuisabel-gif/ros2bag_converter), run as a subprocess and found
via $ROBOTICS_TOOLS_ROS2BAG.  Its JSON export is a list of
{timestamp, topic, type, data} records; from_records() turns those into a SensorLog.

Expected topics (AUVSimBench sim_node defaults):
  imu       sensor_msgs/Imu            orientation (wxyz), angular_velocity, linear_acceleration
  dvl       nav_msgs/Odometry          twist.twist.linear = body velocity
  pressure  sensor_msgs/FluidPressure  depth = (p - atm) / (rho * g)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

from .types import InvalidSensorLog, SensorLog

TOPICS = dict(imu="/imu/data", dvl="/dvl/odometry", pressure="/pressure")
ATM_PA, RHO, G = 101325.0, 1025.0, 9.81   # sim.yaml / AUVParams defaults
TIMEOUT_S = 120


def converter() -> str:
    exe = os.environ.get("ROBOTICS_TOOLS_ROS2BAG", "")
    if not exe or not os.path.exists(exe):
        raise FileNotFoundError("set ROBOTICS_TOOLS_ROS2BAG to ros2bag_convert.py")
    return exe


def _stamp(rec) -> float:
    h = rec["data"].get("header", {}).get("stamp")
    if h and (h.get("sec") or h.get("nanosec")):
        return h["sec"] + h["nanosec"] * 1e-9
    return float(rec["timestamp"])          # bag receive time as fallback


def from_records(records, topics=TOPICS, atm_pa=ATM_PA, rho=RHO, g=G) -> SensorLog:
    """Build a SensorLog from the converter's JSON records (unvalidated; call
    types.validate / diagnose on the result)."""
    by = {k: [] for k in topics}
    name_to_key = {v: k for k, v in topics.items()}
    for r in records:
        k = name_to_key.get(r.get("topic"))
        if k is not None:
            by[k].append(r)
    missing = [topics[k] for k, v in by.items() if not v]
    if missing:
        raise InvalidSensorLog(f"no messages on {missing}")
    try:
        imu, dvl, pr = by["imu"], by["dvl"], by["pressure"]
        q = [r["data"]["orientation"] for r in imu]
        w = [r["data"]["angular_velocity"] for r in imu]
        a = [r["data"]["linear_acceleration"] for r in imu]
        v = [r["data"]["twist"]["twist"]["linear"] for r in dvl]
        p = [r["data"]["fluid_pressure"] for r in pr]
    except (KeyError, TypeError) as e:
        raise InvalidSensorLog(f"unexpected message layout: {e}") from None
    xyz = lambda rows: np.array([[d["x"], d["y"], d["z"]] for d in rows], float)  # noqa: E731
    return SensorLog(
        imu_t=np.array([_stamp(r) for r in imu]),
        imu_acc=xyz(a), imu_gyro=xyz(w),
        imu_q=np.array([[d["w"], d["x"], d["y"], d["z"]] for d in q], float),
        dvl_t=np.array([_stamp(r) for r in dvl]), dvl_v=xyz(v),
        depth_t=np.array([_stamp(r) for r in pr]),
        depth=(np.array(p, float) - atm_pa) / (rho * g),
    )


def load(bag_path: str, topics=TOPICS, exe: str | None = None, **kw) -> SensorLog:
    """Export the three sensor topics from a .db3/.mcap bag and build a SensorLog."""
    exe = exe or converter()
    cmd = [sys.executable, "-I", exe, bag_path, "-f", "json"]
    for t in topics.values():
        cmd += ["-t", t]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
    except (subprocess.TimeoutExpired, OSError) as e:
        raise InvalidSensorLog(f"ros2bag_convert failed: {e}") from None
    if out.returncode != 0:
        raise InvalidSensorLog(f"ros2bag_convert exit {out.returncode}: {out.stderr.strip()[-300:]}")
    try:
        records = json.loads(out.stdout)
    except json.JSONDecodeError as e:
        raise InvalidSensorLog(f"ros2bag_convert produced invalid JSON: {e}") from None
    return from_records(records, topics, **kw)
