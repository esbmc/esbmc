n = 0


def f():
    global n
    n = n + 1
    return 1


try:
    a += f()
except NameError:
    a = f()
assert n == 1
