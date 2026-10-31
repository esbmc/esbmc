def f(v, n):
    if n == 0:
        return v + 1
    return f(v, n - 1)


x = f(True, 1)
y = f("s", 1)
