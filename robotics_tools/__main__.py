"""CLI:  python3 -m robotics_tools demo [--fault F] [--profile P]
        python3 -m robotics_tools eval [--split dev|holdout] [--frameforge] [--verbose]
        python3 -m robotics_tools frameforge [--fault F]
        python3 -m robotics_tools arm [--fault F] [--profile P]
        python3 -m robotics_tools bag BAG.db3|BAG.mcap [--imu T] [--dvl T] [--pressure T]"""
from __future__ import annotations

import argparse
import json

from . import arm_sim, evaluate
from .arm_sim import ArmSecret
from .bag import TOPICS, load
from .diagnose import diagnose, simple_checks
from .sim import PROFILES, Secret, generate

FAULTS = {
    "healthy": lambda p: Secret("healthy", p, 0),
    "mount_x30": lambda p: Secret("mount_rotation", p, 0, dict(axis="x", angle_deg=30.0)),
    "mount_z30": lambda p: Secret("mount_rotation", p, 0, dict(axis="z", angle_deg=30.0)),
    "offset_0.15": lambda p: Secret("timestamp_offset", p, 0, dict(offset_s=0.15)),
    "dropout_3s": lambda p: Secret("dvl_dropout", p, 0, dict(start_s=8.0, duration_s=3.0)),
    "unknown_dvl_scale": lambda p: Secret("unknown", p, 0, dict(kind="dvl_scale", scale=1.3)),
}

ARM_FAULTS = {
    "healthy": lambda p: ArmSecret("healthy", p, 0),
    "enc_j1_0.1": lambda p: ArmSecret("encoder_offset", p, 0, dict(joint=1, offset_rad=0.1)),
    "enc_j2_0.1": lambda p: ArmSecret("encoder_offset", p, 0, dict(joint=2, offset_rad=0.1)),
    "offset_0.1": lambda p: ArmSecret("timestamp_offset", p, 0, dict(offset_s=0.1)),
    "dropout_2s": lambda p: ArmSecret("torque_dropout", p, 0, dict(start_s=5.0, duration_s=2.0)),
    "unknown_payload": lambda p: ArmSecret("unknown", p, 0, dict(kind="payload", mass_kg=0.5)),
}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="robotics_tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--fault", choices=FAULTS, default="mount_x30")
    d.add_argument("--profile", choices=PROFILES, default="hover")
    r = sub.add_parser("arm", help="robot-arm demo (encoders + joint torques)")
    r.add_argument("--fault", choices=ARM_FAULTS, default="enc_j2_0.1")
    r.add_argument("--profile", choices=arm_sim.PROFILES, default="hold")
    e = sub.add_parser("eval")
    e.add_argument("--split", choices=evaluate.SPLITS, default="holdout")
    e.add_argument("--frameforge", action="store_true", help="also run the upstream binary baseline")
    e.add_argument("--verbose", action="store_true")
    f = sub.add_parser("frameforge")
    f.add_argument("--fault", choices=FAULTS, default="mount_x30")
    f.add_argument("--profile", choices=PROFILES, default="hover")
    b = sub.add_parser("bag", help="diagnose a rosbag2 recording (needs $ROBOTICS_TOOLS_ROS2BAG)")
    b.add_argument("path")
    for k, v in TOPICS.items():
        b.add_argument(f"--{k}", default=v)
    a = ap.parse_args(argv)

    if a.cmd in ("demo", "arm"):
        secret = (FAULTS if a.cmd == "demo" else ARM_FAULTS)[a.fault](a.profile)
        res = evaluate.run_case(secret)
        print(f"[evaluator secret, NOT given to diagnostic] {secret}")
        print(f"diagnostic saw {1 + len(res['extra_tests'])} log(s); extra tests requested: {res['extra_tests']}")
        print("checks:", json.dumps(res["checks"], indent=1, default=str))
        print("notes:", res["notes"])
        print("ranking:", res["top1"], "| top3 named:", res["top3"])
        print("simple-checks baseline:", res["baseline"])
    elif a.cmd == "eval":
        results = [evaluate.run_case(s, a.frameforge) for s in evaluate.SPLITS[a.split]()]
        if a.verbose:
            for r in results:
                s = r["secret"]
                print(f"{s.family:17}{s.profile:15}{str(s.params):42} -> {r['top1']:18} extra={r['extra_tests']} base={r['baseline']}"
                      + (f" ff={r['frameforge']}" if "frameforge" in r else ""))
        print(evaluate.report(a.split, results))
    elif a.cmd == "frameforge":
        from .frameforge import binary, frameforge_verdict
        print("binary:", binary())
        log, secret = generate(FAULTS[a.fault](a.profile))
        v = frameforge_verdict(log)
        for sev, fid, summary in v["findings"]:
            print(f"{sev:5} [{fid}] {summary}")
        print("frameforge verdict:", v["verdict"], "| simple-checks:", simple_checks(log),
              "| cross-check top1:", diagnose(log).top1, f"| secret: {secret.family}")
    elif a.cmd == "bag":
        log = load(a.path, {k: getattr(a, k) for k in TOPICS})
        print(f"imu {len(log.imu_t)} dvl {len(log.dvl_t)} depth {len(log.depth_t)} samples")
        d = diagnose(log)
        print("checks:", json.dumps(d.checks, indent=1, default=str))
        print("notes:", d.notes)
        print("ranking:", d.ranking, "| next test:", d.next_test, "| simple-checks:", simple_checks(log))


if __name__ == "__main__":
    main()
