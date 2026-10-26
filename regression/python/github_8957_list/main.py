class M:
    def f(self):
        return 1

    def f2(self):
        return 2


a = M()


def get():
    return a.f2


fs = [a.f, a.f2]
assert fs[0]() == 1
assert fs[1]() == 2
d = {"one": a.f}
assert d["one"]() == 1
g = get()
assert g() == 2
