class A:
    pass

def f(x: A) -> A:
    x.v = 1
    return x

a = A()
b = f(a)

assert b.v == 1
assert a.v == 1
