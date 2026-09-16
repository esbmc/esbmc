class A:
    pass


a = A()
assert type(a) is A
assert a.__class__ is A
assert type(3) is int
assert type(True) is bool
assert type(a) is not int
assert type(A) is type
