# Regression test for issue #3915: a `defaultdict(<factory>)` with no value
# annotation used to default the value type to `char*`, causing
# `min()`/`max()` of a subscript like `d[k]` against an int to abort
# conversion with: "min() arguments must be of comparable types: got
# pointer * subtype: signedbv * width: 8 and signedbv * width: 64".
#
# The dict handler now infers the value type from built-in type factories
# (int / float / bool / str), from the assigned constant's kind, and from
# Lambda factories whose body is a Constant or builtin constructor call.
# Defaultdict reads inside non-Subscript Assign RHS expressions (e.g.,
# `d[k] = min(d[k1], d[k2])`) are also lowered into missing-key init
# statements, so the value-type scan inside nested for-loops still finds
# at least one factory-typed assignment.
#
# Each function uses a unique dict name to avoid a pre-existing
# static-lifetime symbol collision (see the project memory note
# `gotcha_python_symbol_value_static_lifetime`) — unrelated to #3915.
from collections import defaultdict


def f_int():
    a = defaultdict(int)
    a[0, 0] = 5
    return min(a[0, 0], 7)


def f_float():
    b = defaultdict(float)
    b[0, 0] = 5.0
    return min(b[0, 0], 7.0)


def f_literal():
    # No factory; type inferred from the integer literal RHS.
    c = {}
    c[0, 0] = 5
    return min(c[0, 0], 7)


def f_lambda_inf():
    # Nullary Lambda factory returning float('inf') — preprocessor inlines
    # the body so the dict-subscript-assignment is `e[k] = float('inf')`,
    # which the value-type scan recognises as float.
    e = defaultdict(lambda: float("inf"))
    e[0, 0] = 5.0
    return min(e[0, 0], 7.0)


def f_loop_min_rhs():
    # `g[i] = min(g[i], 5)` — the RHS is a Call, not a bare Subscript, so
    # the preprocessor used to skip the missing-key check insertion and
    # the value-type scan saw no factory-typed assignment in the body.
    # With the visit_Assign generic dd_inits lowering, a check is inserted
    # on the read inside min(), giving the scan a `g[i] = int()` to pick
    # up. Exercised inside a for-loop body so the recursive scan walks
    # past the For statement, too.
    g = defaultdict(int)
    g[0] = 7
    for i in range(1):
        g[i] = min(g[i], 5)
    return g[0]


assert f_int() == 5
assert f_float() == 5.0
assert f_literal() == 5
assert f_lambda_inf() == 5.0
assert f_loop_min_rhs() == 5
