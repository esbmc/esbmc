class P:
    def __init__(self):
        self.a = 1


def func(b):
    if b:
        return None
    return P()


p = func(True)
assert p.a == 1
