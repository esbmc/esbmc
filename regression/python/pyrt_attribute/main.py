class A:
    pass


a = A()
a.x = 1
a.x += 1
assert a.x == 2
