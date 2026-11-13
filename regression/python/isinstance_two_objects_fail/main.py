class A:
    pass


class A2:
    pass


c = nondet_bool()
if c:
    a = A()
else:
    a = A2
assert isinstance(a, A)
