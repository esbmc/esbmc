# len() of a computed instance whose class defines no __len__ (#10505)
class C:

    def __init__(self, n: int) -> None:
        self.n = n


def make(n: int) -> C:
    return C(n)


assert len(make(4)) == 0
