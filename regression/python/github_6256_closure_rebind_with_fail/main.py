# A `with ... as` rebinds the enclosing name after the def, so inner() is 4 and
# the claim below is false. The capture cell must not freeze w at its def-time
# 1 -- proving this would be unsound (#6256).


class Ctx:
    def __enter__(self) -> int:
        return 4

    def __exit__(self, a: int, b: int, c: int) -> bool:
        return False


def outer():
    w = 1

    def inner() -> int:
        return w

    with Ctx() as w:
        pass
    return inner


h = outer()
assert h() == 1
