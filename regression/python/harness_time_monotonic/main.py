# Verification harness for time.time monotonicity
# (src/python-frontend/models/time.py).
#
# The model advances a module-level clock by 1.0 on each call, so consecutive
# time() readings strictly increase. This pins that contract; it was left
# unharnessed in PR #5973 under the mistaken belief that the module-global
# clock stalled the converter (it does not — that was machine contention).
#
# ENSURES:
#   E1: each reading is 1.0 greater than the previous one
#   E2: readings strictly increase across three calls
import time

t0: float = time.time()
t1: float = time.time()
t2: float = time.time()

assert t1 == t0 + 1.0       # E1
assert t2 == t1 + 1.0       # E1
assert t0 < t1              # E2
assert t1 < t2              # E2
