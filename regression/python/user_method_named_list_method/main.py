class E:
    def __init__(self):
        self.items = [1, 2]

    def remove(self, pos):
        self.items.pop()

    def append(self, x):
        self.items.append(x + 10)


class B:
    def __init__(self):
        self.e = E()

    def drop(self):
        self.e.remove(0)


e = E()
e.remove(1)
assert len(e.items) == 1

b = B()
b.e.append(5)
assert b.e.items == [1, 2, 15]
b.e.remove(1)
b.drop()
assert b.e.items == [1]

xs = [3]
xs.append(4)
assert xs == [3, 4]
