# Multi-level chained attribute access through None-initialised fields.
# Verifies that the inferred pointer-to-struct type resolves lazily — a
# frozen snapshot of the class layout would break on deeper nesting.

class N:
    def __init__(self, v):
        self.value = v
        self.next = None


a = N(1)
b = N(2)
c = N(3)

a.next = b
b.next = c

assert a.next.next.value == 3
