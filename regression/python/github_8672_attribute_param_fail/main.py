class P:

    def __init__(self, i: int):
        self.ident: int = i


def find(i):
    return "id %d" % i


p = P(3)
assert find(p.ident) == "id 4"
