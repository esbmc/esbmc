# Negative variant: n.next IS None, so asserting it is NOT NoneType must FAIL.
class Node:
    def __init__(self) -> None:
        self.next = None


def main() -> None:
    n = Node()
    assert not isinstance(n.next, type(None))


main()
