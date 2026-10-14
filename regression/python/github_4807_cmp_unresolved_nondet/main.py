# Regression for #4807: `tuple(<list>)` used to reach the comparison
# handler with an unresolved (empty) operand type, so `a == b` silently
# lowered to a nondet bool (this test originally pinned that fallback).
# tuple() over a list is now modelled as the underlying list object, so the
# comparison is a real elementwise equality and can be asserted on.


if __name__ == "__main__":
    a = tuple([1, 2, 3])
    b = tuple([1, 2, 3])
    assert a == b
