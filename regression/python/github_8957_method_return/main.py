class M:
    def s(self):
        return "s"

    def t(self):
        return 3


a = M()


class C:
    def pick(self, flag):
        if flag:
            return a.s
        return a.t


c = C()
flag = nondet_bool()
h = c.pick(flag)
r = h()
if flag:
    assert r == "s"
else:
    assert r == 3
