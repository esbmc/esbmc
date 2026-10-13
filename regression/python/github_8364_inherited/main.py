class A:
    def size(self, k: int) -> int:
        return k

class B(A):
    pass

def size() -> int:
    return 0

b = B()
assert b.size(2) == 2
assert size() == 0
