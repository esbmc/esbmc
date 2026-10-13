class B:
    def __init__(self):
        self.child = self.func2

    def func2(self):
        return 42

    def get(self):
        return self.child

b = B()
f = b.get()
