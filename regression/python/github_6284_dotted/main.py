# github.com/esbmc/esbmc/issues/6284
# A dotted (Attribute) parameter annotation `node: mod.Foo`. The type name is the
# final component (attr = "Foo"), not the module prefix; the annotator must
# dispatch the Attribute shape before the value.id branch, which would otherwise
# infer "mod". The module converts and verifies.
import mod


def use(node: mod.Foo) -> int:
    return node.v


def main() -> None:
    assert use(mod.Foo(42)) == 42


main()
