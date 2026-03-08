from typing import Optional

class List:
    def __init__(self, head: int, tail: Optional["List"]):
        self.head = head
        self.tail = tail

x = List(1, None)
y = List(2, x)

assert y.tail.head == 1
assert y.head == 2
