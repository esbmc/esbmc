# Exercises the --python-irep2-adjust-only sqrt -> ieee_sqrt lowering. A
# non-constant argument makes python_math::handle_sqrt emit a call to
# `c:@F@sqrt` rather than constant-folding; that call must become the ieee_sqrt
# intrinsic, as clang_c_adjust does. Running the library model instead yields
# NaN, so the equality below would fail.
import math


def root(n: int) -> float:
    return math.sqrt(n)


x = 9 if True else 4
assert root(x) == 3.0
