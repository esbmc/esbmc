class Box:

    def __init__(self):
        self.count = 0

    def add(self, n):
        self.count += n


def fill(b, amount):
    b.add(amount)
    assert b.count == 4


b = Box()
fill(b, 5)
