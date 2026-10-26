def f():
    return "s"


def g():
    return 3


class C:
    def pick(self, flag):
        if flag:
            return f
        return g


c = C()
flag = nondet_bool()
h = c.pick(flag)
r = h()
if flag:
    assert r == "s"
else:
    assert r == 3
