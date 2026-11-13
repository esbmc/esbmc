def f(k: int):
    x = 0
    if k == 1:
        x = "a"
    elif k == 2:
        x = 5
    return x
k = nondet_int()
r = f(k)
if k == 1:
    assert r == "a"
elif k == 2:
    assert r == 5
else:
    assert r == 1
