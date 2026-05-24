# Regression test for issue #3915: a `defaultdict(<factory>)` with no value
# annotation used to default the value type to `char*`, causing
# `min()`/`max()` of a subscript like `d[k]` against an int to abort
# conversion with: "min() arguments must be of comparable types: got
# pointer * subtype: signedbv * width: 8 and signedbv * width: 64".
#
# The dict handler now infers the value type from built-in type factories
# (int / float / bool / str), from the assigned constant's kind, and from
# trivial Lambda factories (e.g. `lambda: 0`).
from collections import defaultdict


def f_int():
    d = defaultdict(int)
    d[0, 0] = 5
    return min(d[0, 0], 7)


def f_float():
    d = defaultdict(float)
    d[0, 0] = 5.0
    return min(d[0, 0], 7.0)


def f_literal():
    # No factory; type inferred from the integer literal RHS.
    d = {}
    d[0, 0] = 5
    return min(d[0, 0], 7)


assert f_int() == 5
assert f_float() == 5.0
assert f_literal() == 5
