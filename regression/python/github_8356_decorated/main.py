class V:
    @staticmethod
    def make(k: int) -> int:
        return k

    build = make

assert V.build(3) == 3
