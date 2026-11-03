def f(x: list[int], y: list[int]):
    return x and y


def g(x: list[int], y: list[int]):
    return x or y


c = nondet_bool()
x: list[int] = [7] if c else []
r = f(x, [1])
s = g(x, [1])
if c:
    assert len(r) == 1 and r[0] == 1
    assert len(s) == 1 and s[0] == 7
else:
    assert len(r) == 0
    assert len(s) == 1 and s[0] == 1
