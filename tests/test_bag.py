"""rosbag2 loader: a synthetic .db3 bag (CDR-encoded by hand) round-trips through
ros2bag_convert.py into a SensorLog that diagnoses identically to the original."""
import json
import os
import sqlite3
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robotics_tools import bag  # noqa: E402
from robotics_tools.diagnose import diagnose  # noqa: E402
from robotics_tools.sim import Secret, generate  # noqa: E402
from robotics_tools.types import InvalidSensorLog  # noqa: E402

CONVERTER = os.environ.get("ROBOTICS_TOOLS_ROS2BAG", "")


# ---- minimal CDR writer (little-endian, 4-byte encapsulation header) ------------------
class CDRW:
    def __init__(self):
        self.b = bytearray(b"\x00\x01\x00\x00")

    def align(self, n):
        rel = (len(self.b) - 4) % n
        if rel:
            self.b += b"\x00" * (n - rel)

    def f64(self, *vals):
        for v in vals:
            self.align(8)
            self.b += struct.pack("<d", v)

    def i32(self, v):
        self.align(4)
        self.b += struct.pack("<i", v)

    def u32(self, v):
        self.align(4)
        self.b += struct.pack("<I", v)

    def s(self, text):
        raw = text.encode() + b"\x00"
        self.u32(len(raw))
        self.b += raw

    def header(self, t, frame="base_link"):
        sec = int(t)
        self.i32(sec)
        self.u32(int(round((t - sec) * 1e9)))
        self.s(frame)


def imu_msg(t, acc, gyro, q):
    w = CDRW(); w.header(t)
    w.f64(q[1], q[2], q[3], q[0]); w.f64(*[0.0] * 9)
    w.f64(*gyro); w.f64(*[0.0] * 9)
    w.f64(*acc); w.f64(*[0.0] * 9)
    return bytes(w.b)


def odom_msg(t, v):
    w = CDRW(); w.header(t, "odom"); w.s("dvl_link")
    w.f64(*[0.0] * 3); w.f64(0.0, 0.0, 0.0, 1.0); w.f64(*[0.0] * 36)
    w.f64(*v); w.f64(*[0.0] * 3); w.f64(*[0.0] * 36)
    return bytes(w.b)


def pressure_msg(t, depth):
    w = CDRW(); w.header(t)
    w.f64(bag.ATM_PA + bag.RHO * bag.G * depth, 0.0)
    return bytes(w.b)


def write_bag(path, log):
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT, type TEXT,
                            serialization_format TEXT, offered_qos_profiles TEXT);
        CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER, data BLOB);
    """)
    conn.executemany("INSERT INTO topics VALUES (?,?,?,?,?)", [
        (1, "/imu/data", "sensor_msgs/msg/Imu", "cdr", ""),
        (2, "/dvl/odometry", "nav_msgs/msg/Odometry", "cdr", ""),
        (3, "/pressure", "sensor_msgs/msg/FluidPressure", "cdr", ""),
    ])
    rows = [(1, t, imu_msg(t, a, g, q)) for t, a, g, q in zip(log.imu_t, log.imu_acc, log.imu_gyro, log.imu_q)]
    rows += [(2, t, odom_msg(t, v)) for t, v in zip(log.dvl_t, log.dvl_v)]
    rows += [(3, t, pressure_msg(t, d)) for t, d in zip(log.depth_t, log.depth)]
    conn.executemany("INSERT INTO messages(topic_id, timestamp, data) VALUES (?,?,?)",
                     [(tid, int(t * 1e9), blob) for tid, t, blob in rows])
    conn.commit(); conn.close()


def records_from(log):
    """The converter's JSON shape, built directly (no converter needed)."""
    xyz = lambda v: dict(x=float(v[0]), y=float(v[1]), z=float(v[2]))  # noqa: E731
    hdr = lambda t: dict(header=dict(stamp=dict(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)))))  # noqa: E731
    out = [dict(timestamp=t, topic="/imu/data", type="sensor_msgs/msg/Imu", data=dict(
        **hdr(t), orientation=dict(w=float(q[0]), x=float(q[1]), y=float(q[2]), z=float(q[3])),
        angular_velocity=xyz(g), linear_acceleration=xyz(a)))
        for t, a, g, q in zip(log.imu_t, log.imu_acc, log.imu_gyro, log.imu_q)]
    out += [dict(timestamp=t, topic="/dvl/odometry", type="nav_msgs/msg/Odometry",
                 data=dict(**hdr(t), twist=dict(twist=dict(linear=xyz(v)))))
            for t, v in zip(log.dvl_t, log.dvl_v)]
    out += [dict(timestamp=t, topic="/pressure", type="sensor_msgs/msg/FluidPressure",
                 data=dict(**hdr(t), fluid_pressure=bag.ATM_PA + bag.RHO * bag.G * float(d)))
            for t, d in zip(log.depth_t, log.depth)]
    return out


SECRET = Secret("mount_rotation", "surge_pulses", 0, dict(axis="x", angle_deg=20.0))


def assert_same(a, b):
    for k in ("imu_t", "imu_acc", "imu_gyro", "imu_q", "dvl_t", "dvl_v", "depth_t", "depth"):
        assert np.allclose(getattr(a, k), getattr(b, k), atol=1e-6), k


def test_from_records_round_trip():
    log, _ = generate(SECRET)
    loaded = bag.from_records(records_from(log))
    assert_same(log, loaded)
    assert diagnose(loaded).ranking == diagnose(log).ranking


def test_from_records_errors():
    log, _ = generate(SECRET)
    recs = records_from(log)
    with pytest.raises(InvalidSensorLog, match="no messages"):
        bag.from_records([r for r in recs if r["topic"] != "/pressure"])
    bad = [dict(r, data={"header": r["data"]["header"]}) for r in recs]
    with pytest.raises(InvalidSensorLog, match="layout"):
        bag.from_records(bad)


def test_missing_converter(monkeypatch, tmp_path):
    monkeypatch.delenv("ROBOTICS_TOOLS_ROS2BAG", raising=False)
    with pytest.raises(FileNotFoundError):
        bag.load(str(tmp_path / "x.db3"))


@pytest.mark.skipif(not CONVERTER, reason="set ROBOTICS_TOOLS_ROS2BAG to ros2bag_convert.py")
def test_db3_round_trip(tmp_path):
    log, _ = generate(SECRET)
    path = tmp_path / "run_0.db3"
    write_bag(str(path), log)
    loaded = bag.load(str(path))
    assert_same(log, loaded)
    assert diagnose(loaded).ranking == diagnose(log).ranking


@pytest.mark.skipif(not CONVERTER, reason="set ROBOTICS_TOOLS_ROS2BAG to ros2bag_convert.py")
def test_db3_not_a_bag(tmp_path):
    p = tmp_path / "junk.db3"
    p.write_bytes(b"not sqlite")
    with pytest.raises(InvalidSensorLog, match="ros2bag_convert"):
        bag.load(str(p))
