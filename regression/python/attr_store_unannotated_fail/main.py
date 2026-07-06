# store(5) sets self.v to 5, not 99.
class Box:
    def __init__(self):
        self.v = 0

    def store(self, val):
        self.v = val


b = Box()
b.store(5)
assert b.v == 99
