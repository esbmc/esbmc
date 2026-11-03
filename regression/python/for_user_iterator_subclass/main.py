# A subclass inherits __iter__ and __next__, so it iterates the same way.
class Count:
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


class Sub(Count):
    pass


def main():
    total = 0
    for x in Sub(3):
        total += x
    assert total == 6


main()
