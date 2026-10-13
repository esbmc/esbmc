class C:

    def __init__(self) -> None:
        self.s: str = "at"


def g() -> str:
    return "q"


def f(a: str) -> str:
    x = a + "!"
    c = C()
    return ",".join([a, x, g(), c.s])


assert f("xy") == ",xy!,q,at"
