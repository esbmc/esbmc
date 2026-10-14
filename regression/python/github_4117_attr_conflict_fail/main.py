# Negative variant of github_4117_attr_conflict: an attribute assigned values
# of two classes resolves to the LAST write (flow-sensitive, #4771). The last
# write is `n1.next = o` (Other, value 99), so n1.next.value is 99 — NOT n2's
# value 2. Asserting 2 must be reported as a violation: this proves the read
# uses the last-write class, not the first, and is not a vacuous pass.


class Node:
    def __init__(self, v):
        self.value = v
        self.next = None


class Other:
    def __init__(self, v):
        self.value = v


n1 = Node(1)
n2 = Node(2)
o = Other(99)

n1.next = n2
n1.next = o     # last write wins: n1.next is the Other instance (value 99)

assert n1.next.value == 2
