# The defaulted container is really measured, not assumed: an omitted argument
# yields an empty list, so asserting otherwise is detected.


class Node:
    def __init__(self, value, successors=[]):
        self.value = value
        self.successors = successors


n1 = Node(1)
assert len(n1.successors) == 1
