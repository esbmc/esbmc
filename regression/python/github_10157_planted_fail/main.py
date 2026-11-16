# Issue #10157: the left operand is read before the statements the right one
# plants, not between those and its inline operator call.
i = 0


class B:

    def __add__(self, o: int) -> int:
        global i
        i = i + 2
        return o


class C:

    def __add__(self, o: int) -> int:
        global i
        i = i - 2
        return o


b = B()
c = C()
r = i == [b + 0][0] + (c + 0)
assert not r
