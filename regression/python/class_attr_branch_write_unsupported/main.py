class C:
    v = 0


C.v = 2
a = C()
if nondet_bool():
    a.v = 5
assert a.v == 2 or a.v == 5
