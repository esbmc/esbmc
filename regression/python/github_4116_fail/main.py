# Negative counterpart of github #4116: ensure the coercion doesn't turn the
# comparison into a tautology. After closing the cycle, a.next.next.next is b
# (not a), so the assertion must fail.

class Node:
    def __init__(self):
        self.next = None


a = Node()
b = Node()

a.next = b
b.next = a

assert a.next.next.next == a
