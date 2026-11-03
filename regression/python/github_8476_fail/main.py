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


c = C(3)
t = 0
for i in c:
    t += i
assert t == 7
