import math
from typing import Any


# 1) dist with deep nested aliases on both sides.
p_base = [1.0, [2.0, [3.0, [complex(1.0, 0.0)]]]]
q_base = [1.0, [2.0, [3.0, [4.0]]]]
p_alias = p_base
q_alias = q_base
raised = False
try:
    math.dist(p_alias, q_alias)  # type: ignore
except TypeError:
    raised = True
assert raised


# 2) isclose kwargs merge with overwrite on b plus rel_tol/abs_tol.
kw0 = {"a": 1.0, "b": 1.0, "rel_tol": 1e-9}
kw1 = {"abs_tol": 0.0}
kw2 = {"b": complex(0.0, 1.0)}
raised = False
try:
    math.isclose(**kw0, **kw1, **kw2)  # type: ignore
except TypeError:
    raised = True
assert raised


# 3) fsum with globals + function-returned values.
global_vals = [1.0, 2.0]


def from_fn() -> list[Any]:
    return [3.0, complex(1.0, 0.0)]


combined = global_vals + from_fn()
raised = False
try:
    math.fsum(combined)  # type: ignore
except TypeError:
    raised = True
assert raised




# 5) Callee local-scope resolution for complex return.
def ret_complex_local() -> complex:
    local_z = complex(2.0, 0.0)
    return local_z


raised = False
try:
    math.isclose(1.0, ret_complex_local())  # type: ignore
except TypeError:
    raised = True
assert raised


# 6) One-arg conversion exception must win over guard TypeError.
raised_value_error = False
try:
    math.sin(complex("bad"))
except ValueError:
    raised_value_error = True
assert raised_value_error

# 4) Dedicated cpp-throw priority checks for guarded two-arg math funcs.
raised_value_error = False
try:
    math.remainder(1.0, complex("bad"))
except ValueError:
    raised_value_error = True
assert raised_value_error

raised_value_error = False
try:
    math.ldexp(complex("bad"), 2)
except ValueError:
    raised_value_error = True
assert raised_value_error

raised_value_error = False
try:
    math.nextafter(complex("bad"), 1.0)
except ValueError:
    raised_value_error = True
assert raised_value_error

# 7) Deep static nesting should still be guarded.
raised = False
try:
    math.fsum([[[[[[[[[[[[[[[[[[[[complex(1.0, 0.0)]]]]]]]]]]]]]]]]]]]])  # type: ignore
except TypeError:
    raised = True
assert raised
