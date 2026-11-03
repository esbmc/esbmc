class V:
    def scale(self, k: int) -> int:
        return k

    times = twice = scale

assert V().times(1) == 1
assert V().twice(1) == 1
