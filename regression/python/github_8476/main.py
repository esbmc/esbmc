class C:

    def __init__(self, m: int):
        self.m = m
        self.n = 0

    def __iter__(self):
        return self

    def __next__(self) -> int:
        if self.n >= self.m:
            raise StopIteration
        self.n += 1
        return self.n


def total(m: int) -> int:
    it = C(m)
    s = 0
    for i in it:
        s += i
    return s


c = C(2)
xs = [i for i in c]
assert xs == [1, 2]
assert total(3) == 6
