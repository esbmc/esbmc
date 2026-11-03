class P:
    def size(self, k: int) -> int:
        return k

def size(a: int, b: int) -> int:
    return a + b

p = P()
assert p.size(1, 2) == 3
