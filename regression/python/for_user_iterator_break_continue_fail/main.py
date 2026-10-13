# break and continue in a for over a user iterator keep their meaning.
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
    for x in Count(5):
        if x == 2:
            continue
        if x == 4:
            break
        total += x
    assert total == 5


main()
