from typing import Optional


class Ex:

    def shutdown(self) -> int:
        return 1


def up(name: Optional[str]) -> str:
    if name is None:
        return ""
    return name.upper()


def low(s: str) -> str:
    return s.lower()


def main() -> None:
    x = None
    if x is not None:
        x.foo()
    try:
        x.bit_length()
        assert False
    except AttributeError:
        pass
    e: Optional[Ex] = None
    if nondet_bool():
        e = Ex()
    if e is not None:
        assert e.shutdown() == 1
    try:
        e.shutdown()
    except AttributeError:
        assert e is None
    assert up(None) == ""
    assert up("ab") == "AB"
    assert low("AB") == "ab"
    t = "Hi"
    assert t.lower() == "hi"


main()
