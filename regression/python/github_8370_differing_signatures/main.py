def g(x: int) -> int:
    return x + 1

def h(x: float) -> float:
    return x * 2

class B:
    def __init__(self):
        self.child = g

    def swap(self):
        self.child = h

b = B()
assert b.child(3) == 4
