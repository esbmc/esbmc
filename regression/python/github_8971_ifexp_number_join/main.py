def is_five(v):
    return v == 5


use_float = nondet_bool()
x = 2.5 if use_float else 5
assert is_five(x) == (not use_float)
