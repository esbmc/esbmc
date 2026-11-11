# Issue #10157: the left operand is read before the right one's __add__ runs.
i = 0


class A:

    def __add__(self, o: int) -> int:
        global i
        i = i + 2
        return 1 + o


a = A()
r = i < a + 0
assert r
assert i < a + 2
