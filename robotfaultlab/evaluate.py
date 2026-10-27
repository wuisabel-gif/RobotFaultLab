"""Blinded evaluator.  Holds Secrets; hands the diagnostic SensorLogs only.

Protocol per case:
  1. generate(secret) -> log; diagnose([log])
  2. while diagnosis.next_test and extra tests < MAX_EXTRA: generate the same
     secret fault under the requested motion profile, diagnose(all logs)
Scoring is at fault-family level.  Denominators are ALL cases of a family,
including those where the diagnostic abstained (abstention = not correct for
known families; = correct for "unknown" truth).  top3 is over the 6 named
hypotheses (+abstain), i.e. not a tautology over 3 candidates, but note that
at family level there are only 4 known families.
"""
from __future__ import annotations

from collections import defaultdict

from .diagnose import ABSTAIN, diagnose, family, simple_checks
from .sim import Secret, generate

MAX_EXTRA = 2


def _mount(axis, deg, prof, seed):
    return Secret("mount_rotation", prof, seed, dict(axis=axis, angle_deg=deg))


def _offset(s, prof, seed):
    return Secret("timestamp_offset", prof, seed, dict(offset_s=s))


def _dropout(dur, prof, seed):
    return Secret("dvl_dropout", prof, seed, dict(start_s=8.0, duration_s=dur))


def dev_cases() -> list[Secret]:
    """Used to set thresholds by hand.  Magnitudes: 20 deg, 0.2 s, 3 s."""
    out = []
    for prof in ("forward_cruise", "surge_pulses"):
        for seed in (0, 1):
            out.append(Secret("healthy", prof, seed))
            out += [_mount(ax, 20.0, prof, seed) for ax in "xyz"]
            out.append(_offset(0.2, prof, seed))
            out.append(_dropout(3.0, prof, seed))
    out.append(Secret("healthy", "hover", 0))
    out.append(_dropout(3.0, "hover", 0))
    return out


def holdout_cases() -> list[Secret]:
    """Held-out magnitudes AND motion profiles, plus unknown faults not in the hypothesis set."""
    out = []
    for prof in ("yaw_sweep", "depth_bob", "random_wrench"):
        for seed in (10, 11):
            out.append(Secret("healthy", prof, seed))
        for ax in "xyz":
            for deg in (8.0, 35.0, 60.0):
                out.append(_mount(ax, deg, prof, 10))
        for s in (0.06, 0.12, 0.5):
            out.append(_offset(s, prof, 10))
        for dur in (1.0, 6.0):
            out.append(_dropout(dur, prof, 10))
        out.append(Secret("unknown", prof, 10, dict(kind="dvl_scale", scale=1.3)))
        out.append(Secret("unknown", prof, 10, dict(kind="accel_bias", bias_mps2=0.6)))
    return out


SPLITS = {"dev": dev_cases, "holdout": holdout_cases}


def run_case(secret: Secret, with_frameforge=False) -> dict:
    log, _ = generate(secret)
    logs = [log]
    extra = []
    diag = diagnose(logs)
    while diag.next_test and len(extra) < MAX_EXTRA:
        extra.append(diag.next_test)
        new_log, _ = generate(Secret(secret.family, diag.next_test, secret.seed + 100, secret.params))
        logs.append(new_log)
        diag = diagnose(logs)
    named = [h for h, _ in diag.ranking if h != ABSTAIN]
    res = dict(secret=secret, top1=diag.top1, top3=named[:3], abstained=diag.top1 == ABSTAIN,
               extra_tests=extra, baseline=simple_checks(log), checks=diag.checks, notes=diag.notes)
    if with_frameforge:
        from .frameforge import frameforge_verdict
        res["frameforge"] = frameforge_verdict(log)["verdict"]
    return res


def score(results: list[dict]) -> dict:
    fam = defaultdict(lambda: dict(n=0, top1=0, top3=0, abstain=0, extra=0, base=0, ff=0, ff_n=0, ff_err=0))
    for r in results:
        truth = r["secret"].family
        f = fam[truth]
        f["n"] += 1
        f["abstain"] += r["abstained"]
        f["extra"] += len(r["extra_tests"])
        f["ff_err"] += r.get("frameforge") == "error"
        if truth == "unknown":
            f["top1"] += r["abstained"]
            f["top3"] += r["abstained"]
            f["base"] += r["baseline"] == "unknown"  # baseline never abstains -> always 0
            if "frameforge" in r:
                f["ff_n"] += 1
        else:
            f["top1"] += (not r["abstained"]) and family(r["top1"]) == truth
            f["top3"] += (not r["abstained"]) and truth in {family(h) for h in r["top3"]}
            f["base"] += r["baseline"] == truth
            if "frameforge" in r:
                f["ff_n"] += 1
                f["ff"] += r["frameforge"] == truth
    healthy_fa = sum(1 for r in results if r["secret"].family == "healthy"
                     and not r["abstained"] and r["top1"] != "healthy")
    return dict(per_family=dict(fam), healthy_false_alarms=healthy_fa,
                total=len(results), total_abstain=sum(r["abstained"] for r in results),
                total_extra=sum(len(r["extra_tests"]) for r in results))


def report(split: str, results: list[dict]) -> str:
    s = score(results)
    lines = [f"split={split}  cases={s['total']}  abstentions={s['total_abstain']}  "
             f"extra_tests_requested={s['total_extra']}  healthy_false_alarms={s['healthy_false_alarms']}",
             f"{'family':18}{'n':>4}{'top1':>8}{'top3':>8}{'abstain':>9}{'extra':>7}{'simple':>8}{'frameforge':>12}"]
    for fam, f in s["per_family"].items():
        ff = f"{f['ff']}/{f['ff_n']}" + (f" ({f['ff_err']} err)" if f["ff_err"] else "") if f["ff_n"] else "-"
        lines.append(f"{fam:18}{f['n']:>4}{f['top1']:>4}/{f['n']:<3}{f['top3']:>4}/{f['n']:<3}"
                     f"{f['abstain']:>7}  {f['extra']:>5}  {f['base']:>3}/{f['n']:<3}{ff:>10}")
    lines.append("top1/top3: correct family at rank 1 / within top 3 of 6 named hypotheses; "
                 "denominators include abstentions. For 'unknown' truth, abstaining is counted correct.")
    lines.append("simple = FrameForge-equivalent gravity-axis + DVL-gap rules; frameforge = the upstream binary itself.")
    return "\n".join(lines)
