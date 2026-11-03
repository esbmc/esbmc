class A:

    def __init__(self, b="s"):
        if isinstance(b, str):
            self.b = b
        else:
            self.B = b


class P:

    def __init__(self, n: int):
        self.k = 0
        if n > 0:
            self.pos = n
        else:
            self.neg = -n


a = A(1)
assert a.B == 1

p = P(3)
assert p.pos == 3
q = P(-2)
assert q.neg == 2
assert q.k == 0
