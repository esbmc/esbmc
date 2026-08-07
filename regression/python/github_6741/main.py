def count(*pos):
    return len(pos)


def total(base, *rest):
    s = base
    for v in rest:
        s = s + v
    return s


def with_default(a, b=10, *rest):
    return a + b + len(rest)


class Acc:
    def __init__(self, start, *more):
        self.value = start
        for m in more:
            self.value = self.value + m


assert count() == 0
assert count(1) == 1
assert count(1, 2, 3) == 3
assert total(1) == 1
assert total(1, 2, 3) == 6
assert with_default(1) == 11
assert with_default(1, 2) == 3
assert with_default(1, 2, 3, 4) == 5
assert Acc(5).value == 5
assert Acc(5, 1, 2).value == 8
