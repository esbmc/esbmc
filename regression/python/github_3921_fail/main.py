class D1:
    def __init__(self):
        self.v = 1

    def __add__(self, other):
        return self.v + other.v

class D2:
    def __init__(self):
        self.w = 5

    def __add__(self, other):
        return self.w + other.w

_ = D1() + D1()
_ = D2() + D2()

class E:
    pass

class F:
    def __radd__(self, other):
        return 77

class G:
    def __radd__(self, other):
        return 88

res1 = E() + F()
assert res1 == 77

res2 = E() + G()
assert res2 == 77