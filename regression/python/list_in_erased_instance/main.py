class Node:
    def __init__(self, value, successors=[]):
        self.value = value
        self.successors = successors


def has_seen_successor(seen, start):
    for n in start.successors:
        if n in seen:
            return True
    return False


a = Node(1)
b = Node(2)
start = Node(3, [a])
assert has_seen_successor([a], start)
assert not has_seen_successor([b], start)
