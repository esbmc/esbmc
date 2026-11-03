class C:
    v = 0


flag = True
if flag:
    C.v = "x"
assert C.v == "x"
assert isinstance(C.v, str)
c = C()
assert c.v == "x"

if not flag:
    C.v = 5
assert C.v == "x"
