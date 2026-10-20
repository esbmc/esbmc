class P:

    def __init__(self, i: int, name: str):
        self.ident: int = i
        self.name: str = name


class Q(P):

    def __init__(self, i: int):
        super().__init__(i, "q")


def find(i):
    return "id %d" % i


def label(n, i):
    return "%s=%d" % (n, i)


def main() -> None:
    p = P(3, "p")
    assert find(p.ident) == "id 3"
    assert label(p.name, p.ident) == "p=3"
    q = Q(5)
    assert find(q.ident) == "id 5"


main()
