def f():
    xs = []
    for n in range(2, 3):
        if all(n % x > 0 for x in xs):
            xs.append(n)
    return xs


f()
