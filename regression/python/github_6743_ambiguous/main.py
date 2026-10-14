# Two function-scope classes share a name. A class symbol is keyed by name
# alone, so ESBMC declines rather than answering for the wrong one (#6743).
def f() -> int:
    class Box:
        def value(self) -> int:
            return 3

    return Box().value()


def g() -> int:
    class Box:
        def value(self) -> int:
            return 9

    return Box().value()


assert f() == 3
assert g() == 9
