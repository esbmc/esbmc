class P:
    def __init__(self, u=1):
        self.x = u


n = 0


def make():
    global n
    n += 1
    return P()


class S:
    def make(self, u):
        return P(u)


assert make().x == 1
assert n == 1
y = make().x
assert y == 1 and n == 2
assert S().make(3).x == 3
