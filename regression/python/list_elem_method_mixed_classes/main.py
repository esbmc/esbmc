class P:
    def size(self, k: int) -> int:
        return k


class Q:
    def size(self, k: int) -> int:
        return k + 1


ps = [P(), Q()]
assert ps[1].size(2) == 3
