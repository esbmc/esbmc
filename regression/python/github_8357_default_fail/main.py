# Issue #8357: a default is evaluated in the class body, where `self` is not
# defined, so only the method body is rewritten.
class Outer:

    class Inner:

        def __init__(self, b: int):
            self.b = b

    def make(self, k: int = self.Inner(4).b) -> int:
        return k


assert Outer().make() == 4
