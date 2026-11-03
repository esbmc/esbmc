# GitHub #9015: hasattr on a call's result.
class P:
    def __init__(self):
        self.x = 1


class R:
    def p(self) -> P:
        return P()


assert not hasattr(R().p(), "x")
