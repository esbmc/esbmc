# Issue #9312: box.errors.E has a receiver that is not a plain name.
class Errors:
    E = KeyError


class Box:
    errors = Errors()


def run() -> int:
    box = Box()
    try:
        raise ValueError("v")
    except box.errors.E:
        return 1
    return 0


assert run() == 1
