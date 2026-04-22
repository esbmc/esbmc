class X:
    def __add__(self, other):
        return 10

class Y:
    def __radd__(self, other):
        return 20

res = X() + Y()
assert res == 10