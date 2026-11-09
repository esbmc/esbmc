# int.__gt__ returns NotImplemented for an A, so Python calls A.__lt__ (#10156).
class A:
    def __init__(self) -> None:
        self.n = 0

    def __lt__(self, o: int) -> bool:
        return True

    def __gt__(self, o: int) -> bool:
        self.n = self.n + 1
        return self.n <= 2

    def __eq__(self, o: int) -> bool:
        return o == 3

    def __le__(self, o: int) -> bool:
        return o >= 5


a = A()
r = 3 > a
assert r
assert 3 == a
assert not (4 == a)
assert 7 >= a
assert not (2 >= a)

i = 0
while i < a:
    i = i + 1
assert i == 2
