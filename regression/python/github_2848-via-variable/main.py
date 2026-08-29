from typing import Any


def pick(c: bool) -> Any:
    if c:
        return 10
    return "ten"


def main() -> None:
    # The call site resolves the Any return correctly (see the
    # -direct-call sibling), but binding it to a variable loses that: the
    # variable takes the first return literal's type instead, so the
    # assertion is reported violated even though CPython accepts this.
    v = pick(False)
    assert v == "ten"


main()
