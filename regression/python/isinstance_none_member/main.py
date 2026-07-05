# isinstance(x, type(None)) on a class-instance attribute drives the is-None
# null check in builtins.cpp, now built in IREP2. The operand n.next is a
# member access (the F-P11 deferred-operand shape), so this pins that the
# migration resolves it rather than aborting at construction.
class Node:
    def __init__(self) -> None:
        self.next = None


def main() -> None:
    n = Node()
    assert isinstance(n.next, type(None))


main()
