import math
from typing import Any


def build_isclose_kwargs() -> dict[str, Any]:
    return {"a": 1.0, "b": complex(2.0, 3.0)}


def get_complex_unannotated() -> complex:
    return complex(4.0, 5.0)


def build_remainder_kwargs(flag: bool) -> dict[str, Any]:
    if flag:
        return {"x": complex(1.0, 0.0), "y": 2.0}
    return {"x": 1.0, "y": complex(0.0, 2.0)}


raised = False
try:
    math.isclose(**build_isclose_kwargs())
except TypeError:
    raised = True
assert raised

raised = False
kw_base = {"a": 1.0, "b": complex(2.0, 3.0)}
kw_alias = kw_base
kw_alias2 = kw_alias
try:
    math.isclose(**kw_alias2)
except TypeError:
    raised = True
assert raised

raised = False
try:
    math.degrees(get_complex_unannotated())  # type: ignore
except TypeError:
    raised = True
assert raised

raised = False
try:
    math.fsum([1.0, (2.0, [complex(1.0, 0.0)])])  # type: ignore
except TypeError:
    raised = True
assert raised

raised = False
try:
    math.dist(([1.0, complex(1.0, 0.0)], 0.0), (0.0, 0.0))  # type: ignore
except TypeError:
    raised = True
assert raised

raised = False
try:
    math.remainder(**{"x": 1.0, "y": complex(1.0, 0.0)})
except TypeError:
    raised = True
assert raised

raised = False
try:
    math.remainder(**build_remainder_kwargs(True))  # type: ignore
except TypeError:
    raised = True
assert raised

raised = False
try:
    math.remainder(**build_remainder_kwargs(False))  # type: ignore
except TypeError:
    raised = True
assert raised

global_vals = [1.0, complex(3.0, 0.0)]
raised = False
try:
    math.fsum(global_vals)
except TypeError:
    raised = True
assert raised


def make_vals():
    return [1.0, complex(5.0, 0.0)]


vals_from_fn = make_vals()
raised = False
try:
    math.fsum(vals_from_fn)
except TypeError:
    raised = True
assert raised

# Multiple kwargs unpacking with key overwrite; the final merged "b" is complex.
kw_a = {"a": 1.0}
kw_b = {"b": 2.0}
kw_c = {"b": complex(9.0, 1.0)}
raised = False
try:
    math.isclose(**kw_a, **kw_b, **kw_c)
except TypeError:
    raised = True
assert raised

# Mixed positional and **kwargs path should still reject complex.
kw_with_complex = {"b": complex(1.0, 2.0)}
raised = False
try:
    math.isclose(1.0, **kw_with_complex)
except TypeError:
    raised = True
assert raised

# prod/sumprod compatibility: Python accepts complex for these.
math.prod([1.0, complex(2.0, 3.0)])  # type: ignore

math.sumprod([1.0], [complex(2.0, 3.0)])  # type: ignore

# If argument conversion throws first, propagate that exception instead of
# replacing it with a complex-guard TypeError.
raised_value_error = False
try:
    math.factorial(complex("bad"))
except ValueError:
    raised_value_error = True
assert raised_value_error

raised_value_error = False
try:
    math.isclose(1.0, complex("bad"))
except ValueError:
    raised_value_error = True
assert raised_value_error

# comb path (separate dispatch) should reject complex from aliased values.
comb_n = complex(5.0, 0.0)  # type: ignore
comb_alias = comb_n
raised = False
try:
    math.comb(comb_alias, 2)  # type: ignore
except TypeError:
    raised = True
assert raised

# Category parity: integer-like vs real-like guards must both raise TypeError.
raised = False
try:
    math.factorial(complex(1.0, 0.0))  # integer-like path
except TypeError:
    raised = True
assert raised

# Direct math.isnan/isinf must reject complex too.
raised = False
try:
    math.isnan(complex(1.0, 0.0))
except TypeError:
    raised = True
assert raised

raised = False
try:
    math.isinf(complex(1.0, 0.0))
except TypeError:
    raised = True
assert raised

# Extreme nested iterable path: complex hidden deeply must still be rejected.
deep_vals = [1.0, [2.0, [3.0, [4.0, [5.0, [complex(1.0, 0.0)]]]]]]
raised = False
try:
    math.fsum(deep_vals)  # type: ignore
except TypeError:
    raised = True
assert raised

raised = False
try:
    math.floor(complex(1.0, 0.0))  # real-like path
except TypeError:
    raised = True
assert raised
