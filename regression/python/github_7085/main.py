class C:
    def __init__(self) -> None:
        self.i: int = 0

    def __iter__(self):
        return self

    def __next__(self) -> int:
        if self.i >= 3:
            raise StopIteration
        self.i = self.i + 1
        return self.i


def main():
    n: int = 0
    c = C()
    for v in c:
        n = n + 1
    assert n == 3


main()
