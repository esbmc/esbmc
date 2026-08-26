# An imported class has no call site in its own module, so an unannotated
# list-default parameter had no inferred type and len() on the attribute
# lowered to strlen instead of the list model.
from node import Node


def main():
    empty = Node(1)
    assert len(empty.successors) == 0

    leaf = Node(2)
    root = Node(3, [leaf])
    assert len(root.successors) == 1

    two = Node(4, [leaf, root])
    assert len(two.successors) == 2


main()
