import random


class C:

    def __init__(self):
        self.b = "xy"


o = C()
assert o.b == "xy"
o.b = 10
assert o.b == 10

p = C()
if random.random() < 0.5:
    p.b = 10
assert p.b == "xy" or p.b == 10
