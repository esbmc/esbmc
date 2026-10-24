# github #9090: an Optional[str] assigned nondet_str() on one path
from typing import Optional


def main() -> None:
    text: Optional[str] = None
    if nondet_bool():
        text = nondet_str()
    assert text is not None


main()
