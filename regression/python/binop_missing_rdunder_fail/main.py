class A:
    def __init__(self) -> None:
        self.x = 1


a = A()
try:
    y = 3 // a
except TypeError:
    y = 0
assert y == 1
