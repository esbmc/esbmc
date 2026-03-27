class Box:
    def __init__(self, value: bool):
        self.value = value

    def __bool__(self) -> bool:
        return self.value


def bad() -> int:
    assert False
    return 1


x = 0 if Box(True) else bad()

assert x == 0
