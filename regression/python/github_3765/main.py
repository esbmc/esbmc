class A:

    def __init__(self, x):
        self.x = x


a = [A(1), A(2)]
sorted_a = sorted(a, key=lambda c: c.x)
