class Bag:

    def __init__(self):
        self.n = 2

    def count(self) -> int:
        return self.n

    def upper(self, k: int) -> int:
        return self.n + k


class Box:

    def __init__(self):
        self.bag = Bag()

    def total(self) -> int:
        return self.bag.count()


b = Bag()
assert b.count() == 2
assert b.upper(1) == 3
assert Box().total() == 2
