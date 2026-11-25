class C:

    def __init__(self, x: int) -> None:
        self.x = x

    def __lt__(self, other: "C") -> bool:
        return 10 // self.x < 0


a = C(1)
xs = [3, 1, 2]
xs.sort()
assert xs[0] == 1
assert sorted([2, 1])[0] == 1
assert min([4, 3]) == 3
assert max((4, 3)) == 4
