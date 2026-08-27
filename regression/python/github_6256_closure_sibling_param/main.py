# A sibling helper reusing the captured name as its own parameter binds its own
# storage, not outer's n, so the capture must survive it (#6256).


def outer(n: int):
    def inner() -> int:
        return n

    def other(n: int) -> int:
        return n + 1

    return inner


h = outer(5)
assert h() == 5
