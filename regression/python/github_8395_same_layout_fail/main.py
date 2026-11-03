class A:
    def __init__(self) -> None:
        self.x = 1


class B:
    def __init__(self) -> None:
        self.x = 1


assert isinstance(B(), A)
