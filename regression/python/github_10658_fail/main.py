import random


class C:

    def __init__(self):
        self.b = "xy"


p = C()
if random.random() < 0.5:
    p.b = 10
assert p.b != "xy"
