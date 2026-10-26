class C:
    v = 0


C.v = "x"
assert C.v == 0
