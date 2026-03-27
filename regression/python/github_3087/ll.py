class Foo:

    def __init__(self, t: str) -> None:
        self.t = t


class Bar:

    def __init__(self, t: str) -> None:
        self.t = t


def create(
    s: str,
    t: str,
    b: bool | None = None,
) -> Foo | Bar:
    if s == 'foo':
        return Foo(t)
    elif s == 'bar':
        return Bar(t)
    else:
        raise NotImplementedError("Unknown class")
