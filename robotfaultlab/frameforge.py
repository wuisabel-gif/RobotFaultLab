"""Adapter that runs the UNMODIFIED upstream FrameForge binary on exported CSVs.

Nothing from FrameForge (Apache-2.0, pinned as a submodule) is copied; it is
invoked as an external executable.  Binary lookup order:
  $ROBOTFAULTLAB_FRAMEFORGE, /tmp/robotfaultlab-frameforge/bin/frameforge,
  sources/FrameForge/zig-out/bin/frameforge.

Failure policy: any timeout, unexpected exit code, unparseable output, or a
missing required finding raises FrameForgeError.  The adapter never defaults
to "healthy"; frameforge_verdict() returns verdict="error" in that case.
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .types import SensorLog, validate

_ROOT = Path(__file__).resolve().parents[1]
PROFILE = Path(__file__).with_name("frameforge_profile.yaml")
_LINE = re.compile(r"^(PASS|FAIL|WARN|INFO)\s+\[([^\]]+)\]\s+(.*)$")
_SUMMARY = re.compile(r"^(PASS|FAIL)\s+\d+ pass, \d+ info, \d+ warn, \d+ fail$")
TIMEOUT_S = 30.0
OK_RETURN_CODES = {0, 1}   # src/finding.zig Reporter.finish: 0 = no FAIL, 1 = some FAIL
# Finding ids the mapping needs; absence means the check did not run on our columns.
REQUIRED = {
    "gravity": ({"gravity/direction", "gravity/wrong-axis", "gravity/inverted"},),
    "timestamps": ({"timestamps/monotonic", "timestamps/out-of-order"}, {"timestamps/gaps"}),
}


class FrameForgeError(RuntimeError):
    pass


def binary() -> str | None:
    for c in (os.environ.get("ROBOTFAULTLAB_FRAMEFORGE"), "/tmp/robotfaultlab-frameforge/bin/frameforge",
              _ROOT / "sources" / "FrameForge" / "zig-out" / "bin" / "frameforge"):
        if c and Path(c).is_file() and os.access(c, os.X_OK):
            return str(c)
    return None


def export_csv(log: SensorLog, out_dir: Path) -> tuple[Path, Path]:
    log = validate(log)
    imu = out_dir / "imu.csv"
    dvl = out_dir / "dvl.csv"
    np.savetxt(imu, np.c_[log.imu_t, log.imu_acc], delimiter=",", fmt="%.6f",
               header="t,linear_acceleration.x,linear_acceleration.y,linear_acceleration.z", comments="")
    np.savetxt(dvl, np.c_[log.dvl_t, log.dvl_v], delimiter=",", fmt="%.6f",
               header="t,twist.twist.linear.x,twist.twist.linear.y,twist.twist.linear.z", comments="")
    return imu, dvl


def run(args: list[str], exe: str | None = None, timeout: float = TIMEOUT_S) -> list[tuple[str, str, str]]:
    """Run one FrameForge command; return [(severity, id, summary)] or raise FrameForgeError."""
    exe = exe or binary()
    if exe is None:
        raise FrameForgeError("frameforge binary not found")
    try:
        p = subprocess.run([exe, *args, "--no-color"], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise FrameForgeError(f"frameforge timed out after {timeout}s: {args[0]}") from None
    except OSError as e:
        raise FrameForgeError(f"could not execute frameforge: {e}") from None
    if p.returncode not in OK_RETURN_CODES:
        raise FrameForgeError(f"frameforge {args[0]} exited {p.returncode}: {p.stderr.strip() or p.stdout.strip()}")
    lines = p.stdout.splitlines()
    findings = [m.groups() for m in map(_LINE.match, lines) if m]
    if not findings or not any(_SUMMARY.match(l) for l in lines):
        raise FrameForgeError(f"frameforge {args[0]}: no findings / summary line in output")
    for group in REQUIRED[args[0]]:
        if not group & {fid for _, fid, _ in findings}:
            raise FrameForgeError(f"frameforge {args[0]}: required finding missing ({sorted(group)})")
    return findings


def frameforge_verdict(log: SensorLog, exe: str | None = None) -> dict:
    """Map FrameForge's own gravity + timestamps findings to a fault family.
    On any adapter failure the verdict is "error" with the message in "error"."""
    try:
        with tempfile.TemporaryDirectory() as d:
            imu, dvl = export_csv(log, Path(d))
            g = run(["gravity", str(imu), "--profile", str(PROFILE)], exe)
            t = run(["timestamps", str(dvl), "--profile", str(PROFILE), "--topic", "dvl/odometry"], exe)
    except (FrameForgeError, ValueError) as e:
        return dict(verdict="error", findings=[], error=str(e))
    ids = {fid: sev for sev, fid, _ in g + t}
    if ids.get("gravity/wrong-axis") == "FAIL" or ids.get("gravity/inverted") == "FAIL":
        verdict = "mount_rotation"
    elif ids.get("timestamps/gaps") == "WARN":
        verdict = "dvl_dropout"
    else:
        verdict = "healthy"
    return dict(verdict=verdict, findings=g + t)
