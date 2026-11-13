class A:
    def __init__(self, c):
        if c:
            self.x = 1

    def get(self):
        return self.x


a = A(True)
assert a.get() == 1
assert a.x == 1
b = A(False)
b.x = 3
assert b.get() == 3
