class Cls:
    def __init__(self, m):
        self.m = m

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n > self.m:
            raise StopIteration
        self.n += 1
        return self.n


def collect(c):
    return [i for i in c]


assert collect(Cls(1)) == [1, 2]
