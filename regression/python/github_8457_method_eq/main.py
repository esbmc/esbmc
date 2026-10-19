class V:
    def __init__(self, x: int):
        self.x = x

    def scale(self, k: int) -> int:
        return self.x * k

    def shift(self, k: int) -> int:
        return self.x + k

    def same(self, other: "V") -> bool:
        return self.scale == other.scale


v = V(2)
w = V(2)
u = v
assert v.scale == v.scale
assert v.scale == u.scale
assert v.scale != w.scale
assert v.scale != v.shift
assert v.same(u)
assert not v.same(w)
