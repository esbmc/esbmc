# github.com/esbmc/esbmc/issues/6264
# Regression guard: the str-receiver exclusion must not break genuine
# list.append (local, self-attribute, and nested-list receivers).
class Box:
    def __init__(self):
        self.items = []

    def add(self, v: int) -> None:
        self.items.append(v)


def main() -> None:
    xs = [1, 2]
    xs.append(3)
    assert len(xs) == 3 and xs[2] == 3

    b = Box()
    b.add(7)
    b.add(8)
    assert len(b.items) == 2 and b.items[1] == 8

    m = [[1], [2]]
    m[0].append(9)
    assert len(m[0]) == 2 and m[0][1] == 9


main()
