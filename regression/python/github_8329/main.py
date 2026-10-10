# A for loop over a user iterator goes through __iter__/__next__, not indexing.
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


def main():
    total = 0
    for x in Count(3):
        total += x
    assert total == 6


main()
