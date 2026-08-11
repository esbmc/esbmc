# A name imported into the module namespace must still resolve to the import
# when it is read inside a function body.
from consts import STEP


def advance(x: int) -> int:
    return x + STEP


assert advance(1) == 8
