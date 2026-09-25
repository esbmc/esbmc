# Verification harness for datetime.datetime
# (src/python-frontend/models/datetime.py).
#
# datetime(year, month, day) stores the calendar fields and defaults the time
# fields (hour/minute/second/microsecond) to 0.
#
# REQUIRES:
#   R1: nondet year, and month/day bounded to valid calendar ranges.
#
# ENSURES:
#   E1: the stored year/month/day match the constructor arguments
#   E2: the time fields default to 0
from datetime import datetime

y: int = nondet_int()
mo: int = nondet_int()
d: int = nondet_int()
__ESBMC_assume(1 <= mo <= 12)
__ESBMC_assume(1 <= d <= 31)

dt = datetime(y, mo, d)

assert dt.year == y  # E1
assert dt.month == mo  # E1
assert dt.day == d  # E1
assert dt.hour == 0  # E2
assert dt.minute == 0  # E2
assert dt.second == 0  # E2
assert dt.microsecond == 0  # E2
