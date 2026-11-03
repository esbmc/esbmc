# github #9090: an Optional[str] keeps its type when a str is stored
from typing import Optional


def main() -> None:
    text: Optional[str] = None
    if nondet_bool():
        text = "ab"
    if text is not None:
        assert len(text) == 2
        assert text == "ab"
        assert text.startswith("a")
        u = text + "c"
        assert u == "abc"
    text = "xyz"
    assert text == "xyz"
    assert len(text) == 3


main()
