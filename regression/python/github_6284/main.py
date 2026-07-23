# github.com/esbmc/esbmc/issues/6284
# A parameter with a string (forward-reference) annotation, used in the body,
# used to crash ESBMC (nlohmann type_error.305). It must convert and verify.
class Foo:
    def __init__(self, v: int):
        self.v = v


def use(node: "Foo") -> int:
    x = node
    return x.v


def main() -> None:
    assert use(Foo(42)) == 42


main()
