class Count:

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


def total(it):
    t = 0
    for i in it:
        t += i
    return t


assert total(Count(3)) == 6
assert total(Count(0)) == 0
