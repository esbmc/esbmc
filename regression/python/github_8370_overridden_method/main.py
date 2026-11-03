class B:
    def __init__(self):
        self.child = self.func2

    def func2(self):
        return 42

class D(B):
    def func2(self):
        return 7

d = D()
assert d.child() == 7
