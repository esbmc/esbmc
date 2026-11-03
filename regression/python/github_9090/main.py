# github #9090: an Optional[str] assigned nondet_str() on one path
from typing import Optional


def main() -> None:
    text: Optional[str] = None
    b = nondet_bool()
    if b:
        text = nondet_str()
    assert (text is None) == (not b)


main()
