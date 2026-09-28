# An `except ... as` rebinds the enclosing name after the def, so inner() is the
# caught exception, not the 1 e held at the def. Freezing the capture cell at 1
# would prove this false claim (#6256).


def outer():
    e = 1

    def inner() -> int:
        return e

    try:
        raise ValueError("boom")
    except ValueError as e:
        z = 2
    return inner


h = outer()
assert h() == 1
