# robotics-tools

Which sensor fault does this robot have, from its logs alone? `robotics-tools`
(formerly RobotFaultLab) cross-checks sensor streams against each other, ranks the fault hypotheses,
**abstains** when the data fits none of them, and **names the next motion test
to run** when the evidence is weak. Evaluation is blinded: fault labels and
magnitudes live only in the evaluator, and the boundary is enforced by tests.

Two platforms share the same protocol, scoring and blinding:

| platform | sensors | faults ranked | cross-check |
|---|---|---|---|
| **AUV** | IMU, DVL, depth | IMU mounting rotation (x/y/z), IMU↔DVL timestamp offset or clock drift, DVL dropout, healthy | lag × rotation fit between IMU- and DVL-derived acceleration; depth rate vs DVL; gyro vs orientation |
| **Robot arm** (planar 2-link) | joint encoders, joint torques | encoder zero offset (joint 1/2), encoder↔torque timestamp offset, torque dropout, healthy | lag × joint-offset fit between measured torque and torque predicted from the encoders through the nominal dynamics |

Status: **simulation only.** Nothing here has been run on a real robot; see
[Limitations](#limitations-and-deferred-items). No hardware, control, camera or
firmware code.

## Layout

| file | side | role |
|---|---|---|
| `robotics_tools/types.py` | neutral | `SensorLog` dataclass, `validate()` (drops non-finite rows, rejects empty/short/mis-shaped/non-monotonic input). Imports only numpy. |
| `robotics_tools/sim.py` | evaluator | motion profiles + fault injection on top of **AUVSimBench's `dynamics.py`** (imported from `sources/AUVSimBench`, Apache-2.0, unmodified). Returns `SensorLog` (sensor-only) and `Secret` (truth) as separate objects. |
| `robotics_tools/diagnose.py` | diagnostic | takes `SensorLog`(s) only. Five checks + hand-set rule table → ranking, abstention, next test. Imports `types.py` only; a fresh-process test asserts that importing it loads neither `sim`, `evaluate` nor `auv_sim_bench`. |
| `robotics_tools/frameforge.py` | baseline | exports CSVs and runs the **unmodified upstream FrameForge binary** (`gravity`, `timestamps`). Nothing from FrameForge is copied; it is built and invoked from its own checkout. Timeout, unexpected exit code, empty/malformed output or a missing required finding raise `FrameForgeError`; the verdict is then `error`, never `healthy`. |
| `robotics_tools/frameforge_profile.yaml` | baseline | FrameForge robot profile for the AUVSimBench generic vehicle (9.81 m/s², IMU 100 Hz, DVL 8 Hz). |
| `robotics_tools/evaluate.py` | evaluator | dev/holdout/hard case lists, sequential protocol, scoring, report. |
| `robotics_tools/bag.py` | loader | rosbag2 (`.db3`/`.mcap`) → `SensorLog`. Decoding is delegated to **ros2bag_convert.py** (Apache-2.0, stdlib-only, [wuisabel-gif/ros2bag_converter](https://github.com/wuisabel-gif/ros2bag_converter)) run as a subprocess via `$ROBOTICS_TOOLS_ROS2BAG`; nothing is vendored. Reads `sensor_msgs/Imu`, `nav_msgs/Odometry` (twist = DVL body velocity) and `sensor_msgs/FluidPressure` (depth = (p − atm)/(ρg)), header stamps first, bag receive time as fallback. Imports `types.py` only. |
| `robotics_tools/arm_types.py` | neutral | `ArmLog` (encoders + joint torques), nominal `ArmModel` (URDF-like public parameters), `inverse_dynamics()`, `validate()`. Imports only `types.py`. |
| `robotics_tools/arm_sim.py` | evaluator | new, deliberately small planar 2R arm: analytic joint trajectories, torques from inverse dynamics of the true arm, fault injection. Returns `ArmLog` and `ArmSecret` separately. |
| `robotics_tools/arm_diagnose.py` | diagnostic | takes `ArmLog`(s) only. Torque gap check + joint lag × encoder-offset fit → ranking, abstention, next test. Covered by the same fresh-process import-boundary test. |
| `pyproject.toml` | packaging | `pip install .`, console command `robotics-tools`; numpy is the only runtime dependency. |
| `robotics_tools/__main__.py` | CLI | `demo`, `arm`, `eval`, `frameforge`, `bag`. |
| `tests/test_auv.py` | | ranking on dev cases, determinism, sequential protocol, score denominators, real-binary adapter test (skipped if no binary). |
| `tests/test_bag.py` | | hand-encoded CDR `.db3` bag round-trips through the converter into a `SensorLog` that diagnoses identically to the simulator original; missing topic / layout / converter / non-bag errors. The two converter-backed tests skip without `$ROBOTICS_TOOLS_ROS2BAG`. |
| `tests/test_arm.py` | | arm inverse dynamics against hand-computed gravity moments, dev ranking with recovered offset and lag, static-pose abstain-then-resolve, invalid input, dev split scores. |
| `.github/workflows/tests.yml` | CI | Python 3.12, numpy 2.1.3, pytest 9.1.0; fetches `ros2bag_convert.py` at a pinned commit, runs `pytest tests`. The FrameForge real-binary test skips in CI (no Zig). |
| `tests/test_boundary.py` | | subprocess `sys.modules` boundary, relabeling invariance, non-finite/empty/short/mis-shaped/zero-signal inputs, adapter failure modes via fake executables. |

### What is reused vs new
- **AUVSimBench** (`sources/AUVSimBench/auv_sim_bench/dynamics.py`): 6-DOF physics, quaternion helpers. Sensor sampling in `sim.py` mirrors `sim_node.py`'s IMU/DVL/depth publishing without ROS (rates from `config/sim.yaml`). Its DVL-dropout semantic (drop messages in a window) is reproduced.
- **FrameForge**: invoked externally as the "existing checks" baseline. Its gravity dominant-axis rule and DVL gap rule are also re-expressed in `simple_checks()` so the baseline can be scored without the binary.
- **New** (`diagnose.py`): `imu_dvl_check` — joint time-lag × rotation fit (Kabsch, gravity-anchored) between IMU-derived and DVL-derived body acceleration; `depth_dvl_check` — DVL vertical velocity through IMU attitude vs depth rate; the rule table, abstention gates and next-test rule.

## Getting the code

```bash
git clone --recurse-submodules https://github.com/wuisabel-gif/robotics-tools
```

`sources/AUVSimBench` and `sources/FrameForge` (both Apache-2.0) are git submodules pinned to commits `7003e8d` and `cefa391`. They are pointers to the upstream repositories, not copies: no upstream code is redistributed here. FrameForge is Apache-2.0 as of the pinned commit (the only change since `31a3fb0`, the commit the results were produced with, is the added LICENSE and a README section); it is still built and invoked from its own checkout rather than copied. In an existing clone run `git submodule update --init`.

robotics-tools itself is licensed under Apache-2.0 (see `LICENSE`).

## Run

Requirements: Python ≥ 3.10 with numpy, and pytest for the tests (developed on 3.12 with numpy 2.1.3, pytest 9.1.0).

`pip install .` also works and adds a `robotics-tools` command. An installed copy has the diagnostics, the arm platform and the bag loader; AUV simulation additionally needs AUVSimBench, found through the submodule when run from a clone or through `ROBOTICS_TOOLS_AUVSIMBENCH=/path/to/AUVSimBench` otherwise.

```bash
python3 -m pytest tests/ -q
python3 -m robotics_tools demo --fault mount_z30 --profile hover      # abstains, requests surge_pulses, then resolves
python3 -m robotics_tools arm --fault enc_j2_0.1 --profile hold        # static arm: abstains, requests the best-separating test, then resolves
python3 -m robotics_tools eval --split dev
python3 -m robotics_tools eval --split arm_holdout --verbose
python3 -m robotics_tools eval --split hard --verbose   # realism split, 24/24 but used for development (not blind)
python3 -m robotics_tools eval --split blind --verbose  # second realism split, run once: 29/39
python3 -m robotics_tools eval --split arm_hard --verbose  # arm with friction / backlash, run once: 15/30
python3 -m robotics_tools eval --split holdout --frameforge --verbose  # ~55 s; needs the frameforge binary
python3 -m robotics_tools frameforge --fault dropout_3s --profile surge_pulses
ROBOTICS_TOOLS_ROS2BAG=/path/to/ros2bag_convert.py python3 -m robotics_tools bag run_0.db3   # real recording; --imu/--dvl/--pressure override topics
```

FrameForge binary lookup: `$ROBOTICS_TOOLS_FRAMEFORGE`, then `/tmp/robotics_tools-frameforge/bin/frameforge`, then `sources/FrameForge/zig-out/bin/frameforge`. The real-binary test is skipped if none exists; the fake-executable failure tests always run. To build the binary without touching the upstream tree, point every Zig cache outside it:

```bash
cd sources/FrameForge && zig build --prefix /path/out --cache-dir /path/cache --global-cache-dir /path/global
export ROBOTICS_TOOLS_FRAMEFORGE=/path/out/bin/frameforge
```

## Protocol and blinding
1. Evaluator builds a `Secret` (family, profile, seed, params) and calls `generate()` → `SensorLog` with only `imu_t, imu_acc, imu_gyro, imu_q, dvl_t, dvl_v, depth_t, depth`.
2. `diagnose([log])` returns a ranking over 6 named hypotheses (`healthy, mount_rotation_{x,y,z}, timestamp_offset, dvl_dropout`), possibly headed by `abstain`, and an optional `next_test`.
3. `next_test` is chosen by `choose_test()`: for each test in a public catalogue the diagnostic predicts the evidence each candidate hypothesis would produce and requests the test with the largest separation between the leading candidates (or between all pairs when nothing is conclusive yet). If `next_test` is set, the evaluator regenerates the **same hidden fault** under that motion profile (new seed) and calls `diagnose(all_logs)`; at most 2 extra tests.
4. Scoring is at family level (`mount_rotation_*` → `mount_rotation`). Denominators are all cases of a family **including abstentions**. Abstaining on a known fault counts as wrong; abstaining on `unknown` truth counts as correct. top3 is over the 6 named hypotheses, not over 3 candidates; at family level only 4 known families exist, so top3 is weak evidence and top1 is the number that matters.

The arm platform runs the identical loop with `ArmSecret` → `ArmLog` (`enc_t, enc_q, tau_t, tau`), 5 named hypotheses (`healthy, encoder_offset_j{1,2}, timestamp_offset, torque_dropout`) and joint-level names collapsed to `encoder_offset` for scoring.

Thresholds in `diagnose.TH` were set by hand from the **dev** split only (20° mounts, 0.2 s offset, 3 s dropout; `forward_cruise`, `surge_pulses`, `hover`). The holdout split was run once after they were frozen.

## Results (not independently verified)

Test suite: 53 passed, 1 skipped (FrameForge binary not built) with `ROBOTICS_TOOLS_ROS2BAG` set (`python3 -m pytest tests/ -q`).

### AUV

Dev split (26 cases): 26/26 top1, 0 abstentions, 0 healthy false alarms, 13 extra tests.

Holdout split — held-out magnitudes (8°/35°/60°, 0.06/0.12/0.5 s, 1 s/6 s), held-out profiles (`yaw_sweep`, `depth_bob`, `random_wrench`), and two fault kinds outside the hypothesis set (DVL scale ×1.3, accelerometer x-bias 0.6 m/s²):

```
split=holdout  cases=54  abstentions=6  extra_tests_requested=32  healthy_false_alarms=0
family               n    top1    top3  abstain  extra  simple  frameforge
healthy              6   6/6     6/6        0      4    6/6           -
mount_rotation      27  27/27   27/27       0     20    6/27          -
timestamp_offset     9   9/9     9/9        0      6    0/9           -
dvl_dropout          6   6/6     6/6        0      0    6/6           -
unknown              6   6/6     6/6        6      2    0/6           -
```
`simple` = FrameForge-equivalent gravity-axis + DVL-gap rules; `frameforge` = the upstream binary on exported CSVs. Both catch dropouts and only the 60° roll/pitch mounts (gravity leaves the z-axis dominant below 45°); neither can see yaw mounts, timestamp offsets, or abstain. The cross-check resolves all of them here, with 0.59 extra tests per case on average. (The holdout has been rerun after each later change to the diagnostic and has stayed at 54/54; it was blind only for the first run.)

Hard split (2026-10-30) — sensor realism the thresholds were never tuned on: a vehicle that holds an ~18° pitch trim (`trim_nm=0.9`, a vehicle property that persists across requested extra tests; violates the level-at-rest prior), DVL beam loss as NaN velocities instead of missing messages, IMU clock *rate drift* (0.5 % and 2 %) instead of a fixed offset, gyro x-bias 0.05 rad/s, and sparse random DVL NaNs (30 %).

First run, thresholds frozen: **13/24** (clock drift 0/6 all abstained; gyro bias 0/3 healthy; sparse NaNs 2/3; one trimmed-vehicle dropout called a pitch mount; the trimmed cases that passed did so only because the then-profile-based trim vanished in the requested extra test).

After the 2026-10-31 fixes (`feat/drift-and-trim-fixes`; the hard split was used to develop them, so it is **no longer a blind holdout** — dev and holdout were rerun and are unchanged at 26/26 and 54/54):

```
split=hard  cases=24  abstentions=6  extra_tests_requested=11  healthy_false_alarms=0
family               n    top1    top3  abstain  extra  simple  frameforge
healthy              6   6/6     6/6        0      4    6/6           -
dvl_dropout          3   3/3     3/3        0      0    3/3           -
timestamp_offset     6   6/6     6/6        0      4    0/6           -
unknown              6   6/6     6/6        6      0    0/6           -
mount_rotation       3   3/3     3/3        0      3    0/3           -
```
24/24 after the gyro check (2026-10-31) and the regularity check and computed test choice (2026-11-24). What changed:
- **Clock drift: 0/6 → 6/6.** When the constant-lag fit cannot close the residual, the lag is refit as `lag + drift·(t − t_mid)` over a coarse drift grid (±0.25 % … ±4 %), then the lag is refined. The fitted drift is reported in the checks and the note.
- **Trimmed vehicle: now resolved without cheating.** When the gravity-anchored fit fails, the same lag/drift is refit *without* the anchor; if that closes, the level-at-rest prior is dropped and noted. A roll/pitch rotation is then checked against the depth cross-check: confirmed if the DVL-through-`imu_q` vertical velocity disagrees with the depth rate, contradicted (vehicle trim, score ×0.3) if it agrees — but only when `depth_sensitivity()` says a mount of that axis and angle *would* have been visible with this motion; otherwise the mount is kept and the ambiguity noted. When the motion so far could not have shown such a mount, the test that would show it is requested.
- **Simulator fix**: the vehicle now starts at 5 m depth. AUVSimBench's default starts at the surface, where `depth = max(0, −z)` clipped during `depth_bob`, so the depth cross-check had been reading mostly clipping (healthy `depth_bob` mismatch 0.31 m/s with 0.04 m/s of real vertical motion). This did not change any dev/holdout verdict but did make the depth check usable.

- **Gyro bias: 0/3 → 3/3 (abstain with reason).** `gyro_orientation_check` compares the gyro with the rate implied by consecutive `imu_q` samples; a mount rotates both streams together and a clock fault shifts both together, so only a bias leaves a steady mean difference (healthy floor ~1e-3 rad/s, gate 0.02). It is not a named hypothesis: the diagnostic abstains and the note names the bias vector. Caveat for real IMUs: if `imu_q` comes from the IMU's own gyro-integrating filter, part of the bias is absorbed into the orientation and this check under-reads it.

- **Sparse beam loss: 2/3 → 3/3 (abstain with reason).** `dvl_gap_check` now reports the share of DVL intervals that skipped a ping; one clean outage gives ~1 %, intermittent loss tens of percent, and above 10 % the diagnostic abstains and says so.
- **Roll mount on a trimmed vehicle: 0/1 → 1/1.** The next test is now computed (see Protocol). A roll mount only shows in the depth cross-check under sway, so `sway_pulses` is requested and confirms it.

A 24/24 on a split that was used to find these fixes says little. The honest measure is the next split.

Blind split (2026-11-24, written after the above and **run once with everything frozen**): new seeds and magnitudes, 12° and 40° mounts, a −3 % clock drift, a 0.1 s offset combined with +1 % drift, a 1.5 s NaN dropout, gyro bias 0.03 rad/s, beam loss at 15 % and 45 %, tripled sensor noise on a healthy vehicle, and a milder pitch trim under `yaw_sweep`:

```
split=blind  cases=39  abstentions=15  extra_tests_requested=19  healthy_false_alarms=0
family               n    top1    top3  abstain  extra  simple  frameforge
healthy              9   6/9     6/9        3      4    9/9           -
mount_rotation      12   9/12   12/12       0     11    1/12          -
timestamp_offset     6   3/6     3/6        3      4    0/6           -
dvl_dropout          3   2/3     3/3        0      0    3/3           -
unknown              9   9/9     9/9        9      0    0/9           -
```
**29/39**, 0 healthy false alarms. All 9 out-of-set faults abstained with the right reason. The 10 misses:
- **Healthy vehicle with 3× sensor noise: 0/3, abstained.** The fit leaves 0.28 m/s² unexplained against a 0.2 gate that was set for 1× noise. No false alarm, but a noisier healthy vehicle is not recognised as healthy. The gate has to scale with measured noise instead of being a constant.
- **Clock drift of −3 %: 0/3, abstained.** The drift grid has 2 % and 4 % and nothing between; neither closes the residual. The mixed offset + 1 % drift cases were 3/3. A continuous drift fit is the fix.
- **12° mounts on a trimmed vehicle under `yaw_sweep`: 0/3, called healthy.** With the gravity anchor dropped, the trim and the mount merge into one fitted rotation; the depth cross-check then agrees with `imu_q` closely enough (0.075 m/s against a 0.1 gate) that the whole rotation is written off as trim. The same mounts were 6/6 on level vehicles and the 40° mount on the trimmed vehicle was caught. Small mounts on a trimmed vehicle are a real blind spot.
- **NaN dropout on the trimmed vehicle: called `mount_rotation_y`** (dropout in the top 3). Same root cause: the trim is read as a pitch mount and outranks the gap evidence.

**Read these numbers in context**: the simulator is clean. Noise is small white Gaussian (accel 0.05 m/s², gyro 0.005 rad/s, DVL 0.01 m/s, depth 0.01 m), there are no biases, scale errors, drift, or latency on the healthy sensors, the vehicle self-rights so "level at rest" holds, and faults are injected one at a time. 100% on 54 cases is a statement about this regime, not about real logs.

### Robot arm

Thresholds in `arm_diagnose.TH` were set on `arm_dev` only (0.1 rad offsets, 0.1 s clock offset, 2 s dropout; `sweep`, `shoulder_only`, `hold`): 22/22 there. The holdout was then run **once** (2026-11-23) and nothing was retuned — held-out magnitudes (0.03/−0.2/0.4 rad, 0.03/0.06/0.25 s, 0.5/4 s), held-out profiles (`elbow_only`, `fast_wave`, `random_smooth`), and two faults outside the hypothesis set (0.5 kg payload at the tip, joint-1 torque-sensor gain ×1.25):

```
split=arm_holdout  cases=45  abstentions=6  extra_tests_requested=1  healthy_false_alarms=0
family               n    top1    top3  abstain  extra  simple  frameforge
healthy              6   6/6     6/6        0      0    6/6           -
encoder_offset      18  15/18   18/18       0      1   12/18          -
timestamp_offset     9   7/9     9/9        0      0    0/9           -
torque_dropout       6   6/6     6/6        0      0    6/6           -
unknown              6   6/6     6/6        6      0    0/6           -
```
40/45 at rank 1, 0 healthy false alarms, all 6 unknown faults abstained. Every named encoder fault named the correct joint, with the offset recovered to within 0.007 rad. `simple` is a single-threshold model-residual monitor plus a gap rule: it flags large encoder offsets but calls every clock offset either healthy or an encoder fault, and calls both unknown faults encoder faults.

The 5 misses, all reported healthy:
- **Joint-2 offset of 0.03 rad (1.7°), 0/3.** The elbow offset only changes a 1.5 N m gravity term, so 0.03 rad moves torque by ~0.04 N m, below the 0.05 N m sensor noise. The fit recovers the offset but the evidence gate rejects it. The same offset on joint 1 (larger moment arm) was caught 3/3.
- **Clock offsets of 0.03 s and 0.06 s under `elbow_only`, 0/2.** One slow joint (1.2 rad/s) changes torque too slowly for a short lag to be visible; the same offsets were caught under `fast_wave` and `random_smooth`.

The arm simulator is new code written for this project, not a pinned upstream: planar 2R rigid links, viscous friction, white noise, prescribed trajectories. It has no joint flexibility, backlash, Coulomb friction, gear ripple or controller, so the 0.07 N m healthy floor is far cleaner than any real arm.

Arm realism split (2026-11-24, **run once**, nothing retuned): the dev faults again, on an arm with 0.2 N m of Coulomb friction that the nominal model does not contain, or with 0.01 rad of gear backlash on the encoders. Both are arm properties and persist through requested extra tests.

```
split=arm_hard  cases=30  abstentions=15  extra_tests_requested=1  healthy_false_alarms=0
family               n    top1    top3  abstain  extra  simple  frameforge
healthy              6   3/6     3/6        3      0    3/6           -
encoder_offset      12   6/12    6/12       6      0    9/12          -
timestamp_offset     6   3/6     3/6        3      1    0/6           -
torque_dropout       6   3/6     3/6        3      0    6/6           -
```
**15/30**, 0 healthy false alarms. The split divides cleanly:
- **Backlash 0.01 rad: 15/15.** It adds a small offset that flips sign with direction; the fit absorbs it and every fault was still named.
- **Coulomb friction 0.2 N m: 0/15, all abstained**, including healthy and both dropouts. Friction that reverses with joint direction cannot be explained by a lag or a constant offset, so 0.2 N m stays unexplained against a 0.12 gate. Abstaining is the designed behaviour for an arm that does not match its model, and nothing was mis-named, but it means the diagnostic is unusable on any arm with unmodelled friction until friction is identified and added to the model. A dropout is also swallowed by the abstention, although the gap evidence does not depend on the model; it should be reported regardless.

The arm holdout was rerun after the next-test choice became computed and is unchanged (40/45, same five misses).

## Limitations and deferred items
- **No real recording has been processed.** Every number here is from simulation. The bag loader exists and is tested on a synthetic bag; running a real log with a deliberately injected fault is the single most valuable missing result, and needs a recording.
- **Known failures from the blind runs** (see Results): constant residual gates do not scale with sensor noise; the clock-drift grid is coarse; small mounts on a trimmed vehicle are written off as trim; unmodelled Coulomb friction makes the arm diagnostic abstain on everything. None of these were patched after the blind runs. Multi-fault cases and coloured noise remain untested.
- **"Level at rest" prior.** The rotation fit is anchored on mean specific force ≈ body +Z (FrameForge's gravity-check assumption). Since the drift/trim fixes the anchor is dropped when it cannot close the fit, and the depth cross-check arbitrates mount vs trim where it is sensitive: pitch under forward/vertical motion, roll only under sway. When the motion cannot show it, the diagnostic now requests the test that can (`sway_pulses` for roll).
- **Gyro term approximation.** The DVL-side acceleration uses `ω × v` with the gyro in the IMU frame (`# ponytail:` in `diagnose.py`); exact only for small mount angles or low angular rates. It did not hurt at 60° under `yaw_sweep` here, but is a known ceiling.
- **Next-test choice is computed, but from a small catalogue.** The diagnostic predicts the evidence each hypothesis would produce under each catalogue test and requests the one that separates the candidates best. The AUV catalogue has three tests with excitation figures measured once on the dev vehicle (`diagnose.TESTS`); the arm catalogue has four trajectories and predicts torques from the nominal model. It is a separation heuristic, not a Bayesian expected-information computation, it cannot design a new motion, and at most two extra tests are requested.
- **Lag grid** is ±1 s at 10 ms steps; drift grid ±0.25 % … ±4 % (searched only when a constant lag fails, on a 50 ms lag grid, then refined). Offsets or drifts outside those are untested. Sign convention: positive = IMU clock ahead.
- **Unknown faults**: only two kinds, both abstained; a fault that happens to be well explained by lag+rotation (e.g. a small accelerometer bias → apparent 3° tilt) would be classified, not abstained.
- **Arm platform is a planar two-link idealisation.** Six- or seven-joint arms, the lag × offset fit in higher dimensions, identifying which of many joints is offset, and real friction are all untested. The diagnostic needs the nominal model (masses, lengths); a wrong model looks like an unknown fault and is abstained on, which is safe but unhelpful. Family-level scoring would count a wrong-joint answer as correct; none occurred, but the table does not show that by itself.
- **Bag loader is AUV-only and untested on real recordings**: `bag.py` has only been exercised on a synthetic `.db3` written from simulator output. Real logs will need their topic names passed on the command line, may carry a non-identity IMU-to-body transform the loader does not apply, and the thresholds were never run on real sensor noise.
- **FrameForge adapter** runs only `gravity` and `timestamps`; `compare` would need two orientation streams the sim does not produce.
- **Input validation is structural only.** `validate()` drops non-finite rows and rejects empty, short, mis-shaped or non-monotonic logs; it does not detect plausible-but-wrong units or clock domains.
- Not independently verified.

## Related projects

By the same author, [wuisabel-gif](https://github.com/wuisabel-gif). The first three are used by this project.

| project | what it is | used here as |
|---|---|---|
| [AUVSimBench](https://github.com/wuisabel-gif/AUVSimBench) | Python ROS 2 physics twin for AUV state-estimation simulation | the AUV dynamics, pinned submodule |
| [FrameForge](https://github.com/wuisabel-gif/FrameForge) | Zig CLI that diagnoses TF tree, frame convention and localization issues in ROS robots | the "existing checks" baseline, pinned submodule |
| [ros2bag_converter](https://github.com/wuisabel-gif/ros2bag_converter) | ROS 2 `.db3`/`.mcap` bags to CSV/JSON with no ROS install | the decoder behind `bag.py` |
| [ThrusterHelper.jl](https://github.com/wuisabel-gif/ThrusterHelper.jl) | Julia toolkit for thruster allocation and control-authority analysis (AUVs, ROVs, spacecraft) | |
| [FlashICP](https://github.com/wuisabel-gif/FlashICP) | CUDA-accelerated ICP for point-cloud registration | |
| [WireDAQ](https://github.com/wuisabel-gif/WireDAQ) | data-acquisition architecture simulator: design before wiring | |
| [PixelLink](https://github.com/wuisabel-gif/PixelLink) | ESP32 + CC1101 radio telemetry with a Python dashboard | |
| [UnderwaterVLA](https://github.com/wuisabel-gif/UnderwaterVLA) | underwater vision-language-action work | |
| [render](https://github.com/wuisabel-gif/render) | CPU renderer for an underwater robotics-pool scene, one Zig file | |

More at [github.com/wuisabel-gif](https://github.com/wuisabel-gif?tab=repositories).
