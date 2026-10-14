# github.com/esbmc/esbmc/issues/6284
# Same string-annotated parameter; the module must convert and its assertion be
# checked (here a wrong one) instead of crashing at the annotation.
class Foo:
    def __init__(self, v: int):
        self.v = v


def use(node: "Foo") -> int:
    x = node
    return x.v


def main() -> None:
    assert use(Foo(42)) == 99


main()
