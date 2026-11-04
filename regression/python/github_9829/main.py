# 9829: an unbound += of a call result under except NameError
def f():
    return 1


try:
    a += f()
except NameError:
    a = f()
assert a == 1
