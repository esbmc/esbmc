class P:
    def get(self, k: int) -> int:
        return k


ps = [P()]
assert ps[0].get(2) == 2
