class A:
    pass
def mk():
    return A()
def func(a):
    return a()
b = func(mk)
assert isinstance(b, A)
