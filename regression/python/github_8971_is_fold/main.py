class A:
    pass


def f():
    return 1


flag = nondet_bool()
x = nondet_int() if flag else nondet_str()
assert x is not None
assert not (x is None)
assert x is not A
assert not (x is f)
assert x is x
assert not (x is not x)
