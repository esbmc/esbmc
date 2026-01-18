def f():
    return [[]]

x = []
x.extend([1] + r for r in f())
