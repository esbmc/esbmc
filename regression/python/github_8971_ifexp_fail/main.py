def is_zero(v):
    return v == 0


use_int = nondet_bool()
x = nondet_int() if use_int else "text"
assert is_zero(x) != (use_int and x == 0)
