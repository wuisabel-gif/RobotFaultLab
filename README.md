# RobotFaultLab — increment 1: cross-check fault disambiguation (simulation only)

Given sensor logs from a simulated AUV (IMU, DVL, depth), rank which fault is
present among **IMU mounting rotation (about x/y/z), IMU↔DVL timestamp offset,
DVL dropout, healthy**, abstain when the data is outside that set, and name the
next motion test to run when the evidence is weak. Evaluation is blinded: fault
labels and magnitudes live only in the evaluator.

Nothing here is hardware, control, camera, pool or firmware work.

## Layout

| file | side | role |
|---|---|---|
| `robotfaultlab/types.py` | neutral | `SensorLog` dataclass, `validate()` (drops non-finite rows, rejects empty/short/mis-shaped/non-monotonic input). Imports only numpy. |
| `robotfaultlab/sim.py` | evaluator | motion profiles + fault injection on top of **AUVSimBench's `dynamics.py`** (imported from `sources/AUVSimBench`, Apache-2.0, unmodified). Returns `SensorLog` (sensor-only) and `Secret` (truth) as separate objects. |
| `robotfaultlab/diagnose.py` | diagnostic | takes `SensorLog`(s) only. Four checks + hand-set rule table → ranking, abstention, next test. Imports `types.py` only; a fresh-process test asserts that importing it loads neither `sim`, `evaluate` nor `auv_sim_bench`. |
| `robotfaultlab/frameforge.py` | baseline | exports CSVs and runs the **unmodified upstream FrameForge binary** (`gravity`, `timestamps`). Nothing from FrameForge is copied (no LICENSE upstream). Timeout, unexpected exit code, empty/malformed output or a missing required finding raise `FrameForgeError`; the verdict is then `error`, never `healthy`. |
| `robotfaultlab/frameforge_profile.yaml` | baseline | FrameForge robot profile for the AUVSimBench generic vehicle (9.81 m/s², IMU 100 Hz, DVL 8 Hz). |
| `robotfaultlab/evaluate.py` | evaluator | dev/holdout/hard case lists, sequential protocol, scoring, report. |
| `robotfaultlab/bag.py` | loader | rosbag2 (`.db3`/`.mcap`) → `SensorLog`. Decoding is delegated to **ros2bag_convert.py** (Apache-2.0, stdlib-only, [wuisabel-gif/ros2bag_converter](https://github.com/wuisabel-gif/ros2bag_converter)) run as a subprocess via `$ROBOTFAULTLAB_ROS2BAG`; nothing is vendored. Reads `sensor_msgs/Imu`, `nav_msgs/Odometry` (twist = DVL body velocity) and `sensor_msgs/FluidPressure` (depth = (p − atm)/(ρg)), header stamps first, bag receive time as fallback. Imports `types.py` only. |
| `robotfaultlab/__main__.py` | CLI | `demo`, `eval`, `frameforge`, `bag`. |
| `tests/test_robotfaultlab.py` | | ranking on dev cases, determinism, sequential protocol, score denominators, real-binary adapter test (skipped if no binary). |
| `tests/test_bag.py` | | hand-encoded CDR `.db3` bag round-trips through the converter into a `SensorLog` that diagnoses identically to the simulator original; missing topic / layout / converter / non-bag errors. The two converter-backed tests skip without `$ROBOTFAULTLAB_ROS2BAG`. |
| `.github/workflows/tests.yml` | CI | Python 3.12, numpy 2.1.3, pytest 9.1.0; fetches `ros2bag_convert.py` at a pinned commit, runs `pytest tests`. The FrameForge real-binary test skips in CI (no Zig). |
| `tests/test_boundary.py` | | subprocess `sys.modules` boundary, relabeling invariance, non-finite/empty/short/mis-shaped/zero-signal inputs, adapter failure modes via fake executables. |

### What is reused vs new
- **AUVSimBench** (`sources/AUVSimBench/auv_sim_bench/dynamics.py`): 6-DOF physics, quaternion helpers. Sensor sampling in `sim.py` mirrors `sim_node.py`'s IMU/DVL/depth publishing without ROS (rates from `config/sim.yaml`). Its DVL-dropout semantic (drop messages in a window) is reproduced.
- **FrameForge**: invoked externally as the "existing checks" baseline. Its gravity dominant-axis rule and DVL gap rule are also re-expressed in `simple_checks()` so the baseline can be scored without the binary.
- **New** (`diagnose.py`): `imu_dvl_check` — joint time-lag × rotation fit (Kabsch, gravity-anchored) between IMU-derived and DVL-derived body acceleration; `depth_dvl_check` — DVL vertical velocity through IMU attitude vs depth rate; the rule table, abstention gates and next-test rule.

## Run

Interpreter: `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3` (CPython 3.12.4, numpy 2.1.3, pytest 9.1.0), which is also what `python3` on PATH resolves to here via `/usr/local/bin/python3`. Homebrew and `/usr/bin/python3` lack numpy. Nothing was installed.

```bash
python3 -m pytest tests/ -q
python3 -m robotfaultlab demo --fault mount_z30 --profile hover      # abstains, requests surge_pulses, then resolves
python3 -m robotfaultlab eval --split dev
python3 -m robotfaultlab eval --split hard --verbose   # realism split; thresholds frozen, 13/24
python3 -m robotfaultlab eval --split holdout --frameforge --verbose  # ~55 s; needs the frameforge binary
python3 -m robotfaultlab frameforge --fault dropout_3s --profile surge_pulses
ROBOTFAULTLAB_ROS2BAG=/path/to/ros2bag_convert.py python3 -m robotfaultlab bag run_0.db3   # real recording; --imu/--dvl/--pressure override topics
```

FrameForge binary lookup: `$ROBOTFAULTLAB_FRAMEFORGE`, then `/tmp/robotfaultlab-frameforge/bin/frameforge`, then `sources/FrameForge/zig-out/bin/frameforge`. The real-binary test is skipped if none exists; the fake-executable failure tests always run. To build the binary without touching the upstream tree, point every Zig cache outside it:

```bash
cd sources/FrameForge && /opt/homebrew/bin/zig build --prefix /path/out --cache-dir /path/cache --global-cache-dir /path/global
export ROBOTFAULTLAB_FRAMEFORGE=/path/out/bin/frameforge
```

## Protocol and blinding
1. Evaluator builds a `Secret` (family, profile, seed, params) and calls `generate()` → `SensorLog` with only `imu_t, imu_acc, imu_gyro, imu_q, dvl_t, dvl_v, depth_t, depth`.
2. `diagnose([log])` returns a ranking over 6 named hypotheses (`healthy, mount_rotation_{x,y,z}, timestamp_offset, dvl_dropout`), possibly headed by `abstain`, and an optional `next_test`.
3. If `next_test` is set, the evaluator regenerates the **same hidden fault** under that motion profile (new seed) and calls `diagnose(all_logs)`; at most 2 extra tests.
4. Scoring is at family level (`mount_rotation_*` → `mount_rotation`). Denominators are all cases of a family **including abstentions**. Abstaining on a known fault counts as wrong; abstaining on `unknown` truth counts as correct. top3 is over the 6 named hypotheses, not over 3 candidates; at family level only 4 known families exist, so top3 is weak evidence and top1 is the number that matters.

Thresholds in `diagnose.TH` were set by hand from the **dev** split only (20° mounts, 0.2 s offset, 3 s dropout; `forward_cruise`, `surge_pulses`, `hover`). The holdout split was run once after they were frozen.

## Results (2026-10-26, after the review corrections, thresholds unchanged; not independently verified)

Test suite: 32 passed, 1 skipped (FrameForge binary not built) with the interpreter above and `ROBOTFAULTLAB_ROS2BAG` set (`python3 -m pytest tests/ -q`).

Dev split (26 cases): 26/26 top1, 0 abstentions, 0 healthy false alarms, 11 extra tests.

Holdout split — held-out magnitudes (8°/35°/60°, 0.06/0.12/0.5 s, 1 s/6 s), held-out profiles (`yaw_sweep`, `depth_bob`, `random_wrench`), and two fault kinds outside the hypothesis set (DVL scale ×1.3, accelerometer x-bias 0.6 m/s²):

```
split=holdout  cases=54  abstentions=6  extra_tests_requested=31  healthy_false_alarms=0
family               n    top1    top3  abstain  extra  simple  frameforge
healthy              6   6/6     6/6        0      4    6/6         6/6
mount_rotation      27  27/27   27/27       0     19    6/27        6/27
timestamp_offset     9   9/9     9/9        0      6    0/9         0/9
dvl_dropout          6   6/6     6/6        0      0    6/6         6/6
unknown              6   6/6     6/6        6      2    0/6         0/6
```
`simple` = FrameForge-equivalent gravity-axis + DVL-gap rules; `frameforge` = the upstream binary on exported CSVs. Both catch dropouts and only the 60° roll/pitch mounts (gravity leaves the z-axis dominant below 45°); neither can see yaw mounts, timestamp offsets, or abstain. The cross-check resolves all of them here, with 0.57 extra tests per case on average.

Hard split (2026-10-30, thresholds still frozen, run once) — sensor realism the thresholds were never tuned on: a vehicle that holds an ~18° pitch trim (`trimmed_hover`, violates the level-at-rest prior), DVL beam loss as NaN velocities instead of missing messages, IMU clock *rate drift* (0.5 % and 2 %) instead of a fixed offset, gyro x-bias 0.05 rad/s, and sparse random DVL NaNs (30 %):

```
split=hard  cases=24  abstentions=8  extra_tests_requested=15  healthy_false_alarms=0
family               n    top1    top3  abstain  extra  simple  frameforge
healthy              6   6/6     6/6        0      4    6/6           -
dvl_dropout          3   2/3     3/3        0      0    3/3           -
timestamp_offset     6   0/6     0/6        6      4    0/6           -
unknown              6   2/6     2/6        2      4    0/6           -
mount_rotation       3   3/3     3/3        0      3    0/3           -
```
13/24 at rank 1. What failed, and why:
- **Clock drift: 0/6, all abstained.** A single lag cannot explain a lag that grows through the run, so the unexplained residual stays above the gate and the diagnostic abstains rather than guessing. Honest but unhelpful; a piecewise or linear-in-time lag model would be the fix.
- **Gyro bias: 0/3, reported healthy.** 0.05 rad/s enters only through `ω × v` and is below the noise floor at these speeds. The diagnostic has no gyro-vs-attitude consistency check, so it cannot see this fault at all.
- **Sparse beam loss: 2/3 abstained, 1 healthy.** After `validate()` drops the NaN rows the DVL is just irregularly sampled; under `random_wrench` the fit still closed and it was called healthy. There is no check on sample-rate regularity.
- **Trimmed vehicle + NaN dropout: called `mount_rotation_y`** (dropout still in top 3). The trim looks like a pitch mount to the gravity anchor, the gap check fired too, and the rule table ranked the mount higher with no extra test requested. Healthy and mounted cases on the same trim were resolved because low excitation triggered a `surge_pulses` request, which the dropout case did not get.
- **NaN dropout under motion: 2/2** and **mounts on a trimmed vehicle: 3/3** worked.

**Read these numbers in context**: the simulator is clean. Noise is small white Gaussian (accel 0.05 m/s², gyro 0.005 rad/s, DVL 0.01 m/s, depth 0.01 m), there are no biases, scale errors, drift, or latency on the healthy sensors, the vehicle self-rights so "level at rest" holds, and faults are injected one at a time. 100% on 54 cases is a statement about this regime, not about real logs.

## Limitations and deferred items
- **Realism holdout is partial.** The hard split covers trim, NaN beam loss, clock drift, gyro bias and sparse NaNs, and shows where the frozen thresholds and the fixed-lag model break (see Results). Still untested: realistic coloured noise, multi-fault cases, accelerometer scale error, DVL bottom-lock loss with altitude. Expect the `unexplained_mps2` gate (0.2 m/s², healthy floor ≈0.09) to need re-tuning with realistic noise.
- **"Level at rest" prior.** The rotation fit is anchored on mean specific force ≈ body +Z, the same assumption FrameForge's gravity check makes. A vehicle that genuinely trims at a roll angle is indistinguishable from a roll-mounted IMU with these sensors. A roll/pitch mount is confirmed independently by the depth cross-check only when there is forward or vertical motion.
- **Gyro term approximation.** The DVL-side acceleration uses `ω × v` with the gyro in the IMU frame (`# ponytail:` in `diagnose.py`); exact only for small mount angles or low angular rates. It did not hurt at 60° under `yaw_sweep` here, but is a known ceiling.
- **Next-test selection is a two-row rule table**, not an information-gain computation: low excitation → `surge_pulses`; mount-vs-offset tie → `depth_bob`. Both extra profiles are fixed; the diagnostic cannot design its own.
- **Lag grid** is ±1 s at 10 ms steps; offsets outside that or DVL rates where 10 ms is coarse are untested. Sign convention: positive = IMU clock ahead.
- **Unknown faults**: only two kinds, both abstained; a fault that happens to be well explained by lag+rotation (e.g. a small accelerometer bias → apparent 3° tilt) would be classified, not abstained.
- **Bag loader is untested on real recordings**: `bag.py` has only been exercised on a synthetic `.db3` written from simulator output. Real logs will need their topic names passed on the command line, may carry a non-identity IMU-to-body transform the loader does not apply, and the thresholds were never run on real sensor noise.
- **FrameForge adapter** runs only `gravity` and `timestamps`; `compare` would need two orientation streams the sim does not produce.
- **Input validation is structural only.** `validate()` drops non-finite rows and rejects empty, short, mis-shaped or non-monotonic logs; it does not detect plausible-but-wrong units or clock domains.
- Not independently verified; no git commit made.
