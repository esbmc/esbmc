class M:
    factor: int = 3

    @classmethod
    def scale(cls, x: int) -> int:
        return x * cls.factor


assert M.scale(4) == 999
