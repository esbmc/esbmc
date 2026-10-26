class M:
    def f(self):
        return 1

    def f2(self):
        return 2


def call(h):
    return h()


a = M()
assert call(a.f2) == 1
