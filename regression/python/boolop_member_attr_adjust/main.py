class Box:
    def __init__(self, a: int, b: int):
        self.a = a
        self.b = b


# BoolOp short-circuit whose operands are member accesses (the F-P11 site the
# V.1k resolve-then-build flip targets). Pins the verdict the flip must preserve.
box = Box(1, 0)
r = box.a and box.b
assert r == 0
s = box.a or box.b
assert s == 1
