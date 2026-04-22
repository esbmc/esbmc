class B:
    def __init__(self):
        self.x = 1

    def __add__(self, other):
        return self.x + other.x

_ = B() + B()

class A:
    pass

class C:
    def __radd__(self, other):
        return 0

res = A() + C()