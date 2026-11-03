def f():
    return 1


def g():
    return 2


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
    assert r == 1
else:
    assert r == 2
