# Cyclic reference: a.next.next == a and a.next.next is a (github #4116).
# `a` is stored by value as tag-Node while `a.next.next` is reached through
# pointer-to-Node fields. The frontend coerces the struct side to its address
# so the comparison matches Python's reference-identity semantics.

class Node:
    def __init__(self):
        self.next = None


a = Node()
b = Node()

a.next = b
b.next = a

assert a.next.next == a
assert a.next.next is a
