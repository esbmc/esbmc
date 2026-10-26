class M:
    def f(self):
        return 1


class Acc:
    def __init__(self, n):
        self.n = n

    def add(self, x):
        return self.n + x


def call(h):
    return h()


def apply(h, v):
    return h(v)


a = M()
acc = Acc(3)
assert call(a.f) == 1
assert apply(acc.add, 4) == 7
fs = [acc.add]
acc.n = 10
assert fs[0](4) == 14
