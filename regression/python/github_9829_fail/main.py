def f():
    return 1


try:
    a += f()
except NameError:
    a = f()
assert a == 2
