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


c = C(2)
xs = [i for i in c]
ys = [i for i in c]
assert xs == [1, 2]
assert ys == []
