# Exercises the --python-irep2-adjust-only `&array` -> `&array[0]` decay. The
# raise sites in the operational models build a struct literal
# `{ .message = &"..." }` whose member is a `char*`; without the decay the
# literal carries a `char(*)[N]` and the member type disagrees with its
# initialiser. Catching the raised ValueError forces that literal to be built.
import math


def safe_sqrt(x: float) -> float:
    try:
        return math.sqrt(x)
    except ValueError:
        return -1.0


assert safe_sqrt(-1.0) == -1.0
