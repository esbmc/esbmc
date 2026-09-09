# A module-level function of the same name cannot be the callee of
# `random.choice(...)` -- the receiver already names the module -- so the
# dispatch must still fire and pick the str variant.
import random


def choice(x: int) -> int:
    return x


def main() -> None:
    c = random.choice("abc")
    assert c == "a" or c == "b" or c == "c"


main()
