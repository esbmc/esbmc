# Issue #8243: a class attribute read off a constructor call.
class C:
    v = 0

    def __init__(self, x: int):
        self.x = x


C.v = 3
assert C(1).v == 3
assert C(2).x == 2
