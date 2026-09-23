# A module-level class and a function-scope class of the same name share one
# symbol just as two function-scope ones do, but the #6765 counter only walked
# function bodies, so g() silently ran the module-level constructor (#7541).
class A:
    def __init__(self) -> None:
        self.x: int = 1


def g() -> int:
    class A:
        def __init__(self) -> None:
            self.x: int = 2

    return A().x


assert g() == 2
