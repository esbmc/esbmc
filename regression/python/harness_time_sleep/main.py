# Verification harness for time.sleep (src/python-frontend/models/time.py).
#
# time.sleep(seconds) models the delay as a no-op guarded by an assertion that
# the delay is non-negative.
#
# REQUIRES:
#   R1: a nondet, non-negative, bounded delay.
#
# ENSURES:
#   E1: a non-negative delay is accepted (the model's assertion holds and
#       control returns normally)
import time

s: float = nondet_float()
__ESBMC_assume(s >= 0.0)
__ESBMC_assume(s <= 1000.0)

time.sleep(s)

assert True     # E1 — reaching here means sleep did not fault
