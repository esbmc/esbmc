class P:
    def size(self, k: int) -> int:
        return k


ps = [P()]
try:
    ps[3].size(2)
    assert False
except IndexError:
    pass
