# Falsification harness for time.sleep (src/python-frontend/models/time.py).
#
# sleep() guards its body with `assert seconds >= 0.0`, so a negative delay
# must trip that assertion.
#
# REQUIRES:
#   R1: a nondet, strictly-negative, bounded delay.
#
# WRONG SETUP (expected to be falsified):
#   F1: time.sleep(s) with s < 0 reaches the model's `assert seconds >= 0.0`.
import time

s: float = nondet_float()
__ESBMC_assume(s < 0.0)
__ESBMC_assume(s >= -1000.0)

time.sleep(s)   # trips the model's assert seconds >= 0.0
