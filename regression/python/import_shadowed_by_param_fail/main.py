from node import Node


def get_value(node):
    return node.value


head = Node(42)

assert get_value(head) == 0
