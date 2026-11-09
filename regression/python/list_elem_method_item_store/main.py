class P:
    def size(self, k: int) -> int:
        return k


class Q:
    def size(self, k: int) -> int:
        return k + 1


ps = [P()]
q = Q()
ps[0] = q
assert ps[0].size(2) == 3
