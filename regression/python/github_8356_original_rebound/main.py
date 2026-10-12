class V:
    def scale(self, k: int) -> int:
        return k

    def scale(self, k: int) -> int:
        return k + 1

    times = scale

assert V().times(1) == 2
