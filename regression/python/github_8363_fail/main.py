# str % with an attribute or call operand lowers like the matching f-string
# (#8363).
class Pkt:
    def __init__(self, i: int):
        self.ident: int = i
        self.name: str = "p"


def one() -> int:
    return 1


def half() -> float:
    return 0.5


p = Pkt(1)
assert "id %d" % p.ident == "id 1"
assert "id %d" % one() == "id 1"
assert "%s=%d, %s" % (p.name, one(), half()) == "p=1, 0.50"
