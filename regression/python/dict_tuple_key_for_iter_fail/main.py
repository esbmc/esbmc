def f():
    d = {(1, 2): 10, (3, 4): 20}
    total = 0
    for u, v in d:
        total = total + u + v
    return total


assert f() == 0
