# Negative variant of github_4117_func_local_attr: a.next.value is b.value == 2,
# not 3. The inferred function-local layout must actually be checked, so the
# wrong assertion is reported as a violation (guards against a vacuous pass /
# nondet field).


class Node:
    def __init__(self, v):
        self.value = v
        self.next = None


def chain() -> int:
    a = Node(1)
    b = Node(2)
    a.next = b
    return a.next.value


assert chain() == 3
