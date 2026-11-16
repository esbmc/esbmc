def inner(v):
    return v


def outer(*args):
    return inner(*args)


flag = nondet_bool()
v = nondet_int() if flag else nondet_str()
assert outer(v) == v
