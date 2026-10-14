# str() on an int constrained to a known value. Exercises the runtime path
# rather than compile-time constant folding.
def __VERIFIER_nondet_int() -> int: ...

x = __VERIFIER_nondet_int()
if x == 42:
    s = str(x)
    assert s == "42"
