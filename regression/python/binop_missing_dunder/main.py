class A:
    def __init__(self) -> None:
        self.x = 1


a = A()
caught = 0
try:
    y = a * 3
except TypeError:
    caught += 1
try:
    z = 2.5 - a
except TypeError:
    caught += 1
try:
    w = a + a
except TypeError:
    caught += 1
assert caught == 3
