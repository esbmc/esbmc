# __iter__ may return a separate iterator object; the loop calls __next__
# on that object, not on the iterable.
class CountIter:
    def __init__(self, n: int):
        self.i = 0
        self.n = n

    def __iter__(self):
        return self

    def __next__(self) -> int:
        if self.i >= self.n:
            raise StopIteration
        self.i += 1
        return self.i


class Counts:
    def __init__(self, n: int):
        self.n = n

    def __iter__(self) -> CountIter:
        return CountIter(self.n)


def main():
    total = 0
    for x in Counts(3):
        total += x
    assert total == 6


main()
