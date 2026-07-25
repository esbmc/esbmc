# github.com/esbmc/esbmc/issues/6284
# Same dotted (Attribute) annotation `node: mod.Foo`; the module converts and its
# assertion (here a wrong one) is checked instead of the annotation being
# mis-resolved to the module prefix.
import mod


def use(node: mod.Foo) -> int:
    return node.v


def main() -> None:
    assert use(mod.Foo(42)) == 99


main()
