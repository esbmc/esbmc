# Issue #8243: writing an attribute off a constructor call must not land on
# the class attribute of the same name, so it is still refused.
class C:
    v = 0

    def __init__(self, x: int):
        self.x = x


C(1).v = 5
assert C(2).v == 0
