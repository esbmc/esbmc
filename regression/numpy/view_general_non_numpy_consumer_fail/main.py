class Box:
    def __init__(self):
        self.shape = 3

    def sum(self):
        return 5

    def copy(self):
        return self.shape + 1

b = Box()
assert b.sum() == 5
c = b.copy()
assert c == 5
