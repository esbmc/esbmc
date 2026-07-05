# Storing an unannotated parameter into an existing integer attribute
# (self.v = val) crashed ESBMC: the parameter is typed as a pointer, and writing
# it into a non-pointer field aborted with2t's type-compat assertion. The RHS is
# now reconciled to the member's integer type.
class Box:
    def __init__(self):
        self.a = 0
        self.b = 0

    def store(self, val):
        self.a = val

    def store_both(self, x, y):
        self.a = x
        self.b = y

    def get(self):
        return self.a


b = Box()
b.store(5)
assert b.a == 5
assert b.get() == 5
b.store(-3)
assert b.a == -3
b.store_both(7, 8)
assert b.a == 7 and b.b == 8
