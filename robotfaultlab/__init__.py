"""RobotFaultLab: cross-check fault disambiguation on simulated AUV sensor logs.

Layout (the blinding boundary is the module boundary):
  sim.py       generates SensorLog (sensor-only) + Secret (truth)  -- evaluator side
  diagnose.py  takes SensorLog(s) only; never imports Secret        -- diagnostic side
  frameforge.py runs the unmodified upstream FrameForge binary as a baseline
  evaluate.py  holds Secrets, scores blinded diagnoses, serves extra tests
"""
