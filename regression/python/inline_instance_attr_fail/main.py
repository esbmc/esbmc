class C:
    def __init__(self, v):
        self.x = v


# C(5).x is 5, not 6.
assert C(5).x == 6
