# __bool__ decides the answer, so asserting the opposite stays refutable.


class Falsy:
    def __bool__(self) -> bool:
        return False


f = Falsy()
assert f
