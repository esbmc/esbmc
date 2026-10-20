# An attribute added from outside on every path reads back as stored.
class A:
    pass


a = A()
a.code = nondet_int()
c = a.code
assert c == a.code
