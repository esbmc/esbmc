class C:
    v = 0


C.v = 2
a = C()
a.v = 5
assert a.v == 5
assert C.v == 2
