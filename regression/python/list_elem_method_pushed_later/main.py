class P:
    def size(self, k: int) -> int:
        return k


class Q:
    def size(self, k: int) -> int:
        return k + 1


ps = [P()]
total = 0
for _ in range(2):
    total += ps[-1].size(2)
    ps.append(Q())
assert total == 5
