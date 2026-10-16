class Count:

    def __init__(self, mx):
        self.mx = mx
        self.n = 0

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n >= self.mx:
            raise StopIteration
        r = self.n
        self.n += 1
        return r


def main():
    c = Count(2)
    it = iter(c)
    assert next(it) == 0
    assert next(it) == 1


main()
