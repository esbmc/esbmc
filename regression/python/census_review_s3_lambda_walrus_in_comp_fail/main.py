def f():
    [(lambda: (divmod := 1)) for _ in ()]
    return divmod(1, 0)


f()
