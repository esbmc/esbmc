class A:
    def __init__(self, c):
        if c:
            self.x = 1

    def get(self):
        return self.x


a = A(False)
y = a.get()
