# Exercises the --python-irep2-adjust-only fabs -> abs lowering. A non-constant
# argument makes the frontend emit a call to `c:@F@fabs`; that call must become
# the abs intrinsic, as clang_c_adjust does, so the hop-off GOTO matches legacy
# (`RETURN: abs(x)`) instead of executing the library model.
import math


def f(x: float) -> float:
    return math.fabs(x)


v = -3.5 if True else 1.0
assert f(v) == 3.5
