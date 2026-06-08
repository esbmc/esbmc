class Node:
    def __init__(self, v):
        self.value = v
        self.next = None

n1 = Node(1)
n2 = Node(2)

n1.next = n2

assert n1.next.value == 2
