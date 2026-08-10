# A class defined inside a method is not supported, but it must not crash:
# the class builder dereferenced the missing method symbol and segfaulted.
# The method stays callable, so the verdict below is still decided.
class Outer:
    def m(self) -> int:
        class Inner:
            pass

        return 1


assert Outer().m() == 1
