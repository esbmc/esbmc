class B:
    def __init__(self):
        self.child = self.func2

    def func2(self):
        return 42

b = B()
assert b.child() == 41
