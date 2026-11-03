def f():
    e = [i * 2 for i in [1.5, 2.5]]
    i = {1: 2.0}
    return i[1]


assert f() == 2.0
