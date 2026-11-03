class V:
    def scale(self, k: int) -> int:
        return k

    times = scale
    r = times(None, 2)


assert V.r == 2
assert V().times(3) == 3
