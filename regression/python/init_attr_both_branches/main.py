class A:
    def __init__(self, c):
        if c:
            self.x = 1
        else:
            self.x = 2

    def get(self):
        return self.x


assert A(True).get() == 1
assert A(False).get() == 2
