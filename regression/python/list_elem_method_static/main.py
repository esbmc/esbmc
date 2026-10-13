class P:
    base = 5

    @staticmethod
    def twice(k: int) -> int:
        return 2 * k

    @classmethod
    def offset(cls, k: int) -> int:
        return cls.base + k


ps = [P()]
assert ps[0].twice(3) == 6
assert ps[0].offset(2) == 7
