def g(x: int) -> int:
    return x + 1

def h(x: int) -> int:
    return x * 2

class B:
    def __init__(self):
        self.child = g

    def swap(self):
        self.child = h

    def run(self, v: int) -> int:
        return self.child(v)

b = B()
assert b.run(3) == 4
b.swap()
assert b.run(3) == 6
