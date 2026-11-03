class A:
    def __init__(self) -> None:
        self.x = 1


class B(A):
    pass


assert isinstance(A(), B)
