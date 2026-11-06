# Calling a parameter named like the class passed for it (#9806).
class A:
    def __init__(self):
        self.v = 1


class B:
    def __init__(self):
        self.v = 2


def f(A):
    x = A()
    return x.v


assert f(A) == 1
assert f(A=B) == 2
