def is_one(v):
    return v == 1


use_int = nondet_bool()
x = 1 if use_int else "text"
assert is_one(x) == use_int
assert is_one(1)
