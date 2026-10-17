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


b = B()
b.e.append(5)
assert b.e.items[2] == 5
