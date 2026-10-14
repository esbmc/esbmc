class P:

    def __init__(self, n: int):
        self.k = 0
        if n > 0:
            self.pos = n
        else:
            self.neg = -n


q = P(-2)
assert q.neg == 3
