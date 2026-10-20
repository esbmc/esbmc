from esbmc import nondet_int, __ESBMC_assume


class Cls:

    def __init__(self, mx):
        self.mx = mx

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n > self.mx:
            raise StopIteration
        r = 2**self.n
        self.n += 1
        return r


m = nondet_int()
__ESBMC_assume(0 <= m and m < 3)
def func(c):
    return [i for i in c]


a = func(Cls(m))
assert len(a) == m + 1
