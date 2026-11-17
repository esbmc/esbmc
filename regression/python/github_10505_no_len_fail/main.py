# #10505: len() of a call result whose class defines no __len__ raises TypeError.
class D:

    def __init__(self) -> None:
        self.n = 1


def make() -> D:
    return D()


len(make())
