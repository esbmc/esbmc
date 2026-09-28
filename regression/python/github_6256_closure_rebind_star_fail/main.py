# A starred unpack target rebinds the enclosing name after the def, so inner()
# is [8, 9], not the 1 s held at the def. Freezing the capture cell at 1 would
# prove this false claim (#6256).


def outer():
    s = 1

    def inner() -> int:
        return s

    head, *s = [7, 8, 9]
    return inner


h = outer()
assert h() == 1
