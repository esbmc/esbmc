class A:
    x = 1

def f(o):
    o.x = 5

a = A()
f(a)
assert a.x == 5
