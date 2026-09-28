# Distinctly named function-scope classes do not collide and must still verify
# (#7541).
def f() -> int:
    class A:
        def __init__(self) -> None:
            self.x = 1
    return A().x


def g() -> int:
    class B:
        def __init__(self) -> None:
            self.x = 2
    return B().x


def main() -> None:
    assert f() == 1
    assert g() == 2


main()
