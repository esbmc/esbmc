# github #9090: index an Optional[str] that held None past its end
from typing import Optional


def main() -> None:
    text: Optional[str] = None
    if nondet_bool():
        text = "ab"
    if text is not None:
        c = text[2]


main()
