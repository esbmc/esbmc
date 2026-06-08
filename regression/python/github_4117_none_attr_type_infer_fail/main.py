class Node:
    def __init__(self, v):
        self.value = v
        self.next = None

n1 = Node(1)
n2 = Node(2)

n1.next = n2

# Wrong expected value: actual is 2, not 99.
assert n1.next.value == 99
