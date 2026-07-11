# Falsification harness for time.time monotonicity
# (src/python-frontend/models/time.py).
#
# The clock advances by 1.0 on each call, so two consecutive readings are never
# equal; asserting equality must be falsifiable.
#
# WRONG PROPERTY (expected to be falsified):
#   F1: two consecutive time() readings are equal.  They differ by 1.0.
import time

t0: float = time.time()
t1: float = time.time()

assert t0 == t1       # F1 — falsifiable (readings differ by 1.0)
