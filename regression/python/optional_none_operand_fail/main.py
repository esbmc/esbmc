from typing import Optional


def main() -> None:
    v: Optional[int] = None
    k: int = 4
    if nondet_bool():
        v = k
    w = v * 2


main()
