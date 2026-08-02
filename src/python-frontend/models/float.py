# pylint: disable=redefined-builtin
# This class intentionally shadows the Python built-in `float`: it is the
# operational model ESBMC uses to verify Python programs, so it must match the
# built-in name exactly. Methods are written as classmethods taking the
# encapsulated value as the first value parameter, mirroring `models/int.py`:
# the frontend lowers a no-argument instance call `x.is_integer()` into
# `is_integer(x)` (see function_call/expr.cpp), passing a null `cls`.
class float:

    @classmethod
    # is_integer() returns True iff the float has no fractional part
    # (e.g. (5.0).is_integer() is True, (5.5).is_integer() is False).
    # int(x) truncates toward zero, so a finite float equals its truncation
    # exactly when it is integral; the comparison is exact for every value in
    # the range ESBMC's bounded checks explore.
    def is_integer(cls, x: float) -> bool:
        return x == int(x)
