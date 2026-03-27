class A:
    x = 1


a = A()
a.x = 2
del a.x

assert a.x == 1
