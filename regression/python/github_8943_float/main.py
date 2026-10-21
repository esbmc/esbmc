class C:
    v = 0


C.v = 2.5
assert C.v == 2.5
c = C()
assert c.v == 2.5
