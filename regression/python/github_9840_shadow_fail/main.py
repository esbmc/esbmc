class S:
    def shadow(self) -> int:
        x = 5
        f = lambda x: x * 2
        return f(3)


assert S().shadow() == 10
