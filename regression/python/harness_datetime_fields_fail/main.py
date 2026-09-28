# Falsification harness for datetime.datetime
# (src/python-frontend/models/datetime.py).
#
# The time fields default to 0, so claiming a non-zero default must be
# falsifiable.
#
# WRONG PROPERTY (expected to be falsified):
#   F1: datetime(y, 6, 15).hour == 1.  The hour defaults to 0.
from datetime import datetime

y: int = nondet_int()
dt = datetime(y, 6, 15)

assert dt.hour == 1  # F1 — falsifiable (hour defaults to 0)
