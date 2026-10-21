class C:
    v = 0


C.v = "x"
assert C.v == "x"
c = C()
assert c.v == "x"
assert isinstance(C.v, str)
