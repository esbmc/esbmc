class A:
    y = 0

    def __init__(self, c):
        if c:
            self.y = 5

    def get(self):
        return self.y


assert A(False).y == 0
assert A(True).y == 5
assert A(False).get() == 0
assert A(True).get() == 5
assert A.y == 0
