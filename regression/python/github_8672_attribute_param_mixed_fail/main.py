class P:

    def __init__(self, i: int):
        self.ident: int = i


def find(i):
    return i


p = P(3)
assert find("x") == "x"
assert find(p.ident) != 3
