class C:
    v = 0


c = C()
flag = True
if flag:
    C.v = "x"
assert c.v == 0
