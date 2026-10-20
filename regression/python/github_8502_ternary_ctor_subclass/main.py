class B:
    def __init__(self, v):
        self.a = v

    def get(self):
        return self.a


class C(B):
    def __init__(self, v):
        self.a = v


b = C(1) if nondet_bool() else B(1)
assert b.get() == 1
