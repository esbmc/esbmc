# Flow-inferred attribute type (None -> Node) under --python-irep2-adjust, the
# github_4117 shape: the adjuster follows the inferred symbol_type tag.
class Node:
    def __init__(self, value: int) -> None:
        self.value: int = value
        self.next = None


def main() -> None:
    n1 = Node(1)
    n2 = Node(2)
    n1.next = n2
    assert n1.next.value == 2


main()
