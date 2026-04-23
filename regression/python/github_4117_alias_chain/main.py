# Alias propagation in the usage-site scanner:
# `alias = original` should carry `original`'s inferred class through
# `alias`, so that `n1.next = alias` is recognised as a class-typed
# assignment.

class Node:
    def __init__(self, v):
        self.value = v
        self.next = None


original = Node(2)
alias = original

n1 = Node(1)
n1.next = alias

assert n1.next.value == 2
