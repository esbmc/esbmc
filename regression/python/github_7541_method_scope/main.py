# The #6765 counter never descended into a ClassDef body, so classes defined
# inside methods were invisible to it and collided silently (#7541).
class P:
    def m(self) -> int:
        class A:
            def __init__(self) -> None:
                self.x: int = 1

        return A().x


class Q:
    def m(self) -> int:
        class A:
            def __init__(self) -> None:
                self.x: int = 2

        return A().x


assert Q().m() == 2
