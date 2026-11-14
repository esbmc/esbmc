def f(k: int):
    x = 0
    if k == 1:
        x = "a"
    elif k == 2:
        x = 5
    return x
assert f(1) == "a"
assert f(2) == 5
assert f(3) == "a"
