class M:
    def f(self):
        return 1

    def f2(self):
        return 2


def call(h):
    return h()


a = M()
flag = nondet_bool()
assert call(a.f if flag else a.f2) == (1 if flag else 2)
i = nondet_int()
__ESBMC_assume(i >= 0 and i < 2)
fs = [a.f, a.f2]
assert fs[i]() == i + 1
