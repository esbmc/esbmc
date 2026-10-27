def outer(v: int) -> int:
    def func(x=None):
        if x is None:
            return 7
        return x
    a = func(v)
    b = func()
    assert b == 7
    return a
v = nondet_int()
assert outer(v) != 0
