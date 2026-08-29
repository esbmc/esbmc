# A break leaving a try/finally runs the finally too, as do returns from an
# except handler and from an else clause. Issue #7076.


class Box:
    def __init__(self) -> None:
        self.v: int = 0


def breaks_out(b: Box) -> int:
    i: int = 0
    while i < 4:
        try:
            if i == 2:
                break
        finally:
            b.v = b.v + 1
        i = i + 1
    return i


def returns_from_handler(b: Box) -> int:
    try:
        raise ValueError("x")
    except ValueError:
        return 3
    finally:
        b.v = 8


def returns_from_else(b: Box) -> int:
    try:
        b.v = 1
    except ValueError:
        b.v = 2
    else:
        return 4
    finally:
        b.v = b.v + 10


def main() -> None:
    b = Box()
    assert breaks_out(b) == 2
    assert b.v == 3

    b2 = Box()
    assert returns_from_handler(b2) == 3
    assert b2.v == 8

    b3 = Box()
    assert returns_from_else(b3) == 4
    assert b3.v == 11


main()
