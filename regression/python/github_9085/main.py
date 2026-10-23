# github #9085: an Optional[str] that may be None passed as an argument
from typing import Optional


def check(text: Optional[str]) -> str:
    if not isinstance(text, str):
        raise TypeError("not a string")
    return text


def main() -> None:
    text: Optional[str] = None
    if nondet_bool():
        text = "a"
    else:
        text = "b"
    check(text)


main()
