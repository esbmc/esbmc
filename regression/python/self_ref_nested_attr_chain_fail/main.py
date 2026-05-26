class Node:
    def __init__(self, value=None, successor=None):
        self.value = value
        self.successor = successor


a = Node(1)
b = Node(2, a)
c = Node(3, b)

assert c.successor.value == 99
