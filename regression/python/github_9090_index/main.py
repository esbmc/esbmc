# github #9090: index and iterate an Optional[str] that held None
from typing import Optional


def main() -> None:
    text: Optional[str] = None
    if nondet_bool():
        text = "ab"
    if text is not None:
        assert text[0] == "a"
        assert text[-1] == "b"
        for c in text:
            assert c == "a" or c == "b"


main()
