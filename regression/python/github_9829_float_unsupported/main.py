def f():
    return 0.5


try:
    a += f()
except NameError:
    a = 1
assert a == 1
