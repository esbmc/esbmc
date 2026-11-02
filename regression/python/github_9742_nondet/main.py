x: list[int] = nondet_list(3, nondet_bool())
y = [1]
r = x and y
if len(x) == 0:
    assert len(r) == 0
else:
    assert r == [1]
s = x or y
if len(x) == 0:
    assert s == [1]
else:
    assert len(s) == len(x)
