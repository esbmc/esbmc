class K:
    def m1(self):
        return 1

    def m2(self):
        return "a"


k = K()
swap = nondet_bool()
if swap:
    c, d = k.m2, k.m1
else:
    c, d = k.m1, k.m2
if swap:
    assert c() == "a"
    assert d() == 1
else:
    assert c() == 1
    assert d() == "a"
