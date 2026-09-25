class A:
    def __init__(self):
        self.x = 1


a = A()
assert hasattr(a, "x")
assert getattr(a, "x") == 1
setattr(a, "y", 2)
assert getattr(a, "y") == 2
assert hasattr(a, "y")
assert not hasattr(a, "z")
