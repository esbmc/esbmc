# Falsification harness for collections.Counter
# (src/python-frontend/models/collections.py).
#
# Validates the missing-key default: an unwritten key must read as 0, never a
# positive count.
#
# WRONG PROPERTY (expected to be falsified):
#   F1: Counter()[(9, 9)] == 1.  False — the model returns 0 for keys that
#       were never written.
from collections import Counter

c: Counter = Counter()

assert c[(9, 9)] == 1       # F1 — falsifiable (unwritten key is 0)
