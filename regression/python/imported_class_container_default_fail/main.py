from node import Node


def main():
    root = Node(1, [Node(2)])
    assert len(root.successors) == 2


main()
