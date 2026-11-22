def inc(x: int) -> int:
    return eval("x + 1")


a = eval("1 + 2")
assert a == 3
assert inc(a) == 4
