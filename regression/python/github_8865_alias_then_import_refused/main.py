class C:

    def __init__(self):
        self.x = 1


D = C
from m2 import C

d = D()
assert d.x == 1
assert C() == 5
