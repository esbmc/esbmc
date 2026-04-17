def f():
    xs: list[int] = []
    for n in range(2, 3):
        if all(n % x > 0 for x in xs):
            xs.append(n)
    return xs


f()
