def is_valid(s: str) -> bool:
    if len(s) < 5 or len(s) > 80:
        return False
    return True


class Foo:

    def __init__(self) -> None:
        pass

    def foo(self, s: str) -> None:
        assert is_valid(s)
