# `for` over a class instance no longer silently skips the body. The loop
# bound is len(obj), and a class with no __len__ has no length, so the run is
# refused instead of proving a dead loop. Once the iterator protocol lands
# this should report the AssertionError from the body instead.

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
    c = C()
    for v in c:
        assert False


main()
