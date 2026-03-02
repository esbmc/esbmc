class A:

    def __init__(self, v):
        self.v = v


a1 = A(1)
a2 = A(2)

assert a1.v + a2.v == 3
