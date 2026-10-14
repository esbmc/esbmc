# An element annotated list[Tuple[...]] used to resolve to an opaque 0-member
# "Tuple" struct (get_typet treated the Tuple[...] Subscript node as a wrapper
# and unwrapped it to the bare name "Tuple"). Reading such an element through a
# function parameter or a function return — e.g. `def pop(xs): return xs[0]` —
# then crashed the backend (value-set member access / SMT cast-to-struct).
# The annotation now resolves to the concrete tuple struct, matching what a
# tuple literal produces, so the element carries real components.
from typing import List, Tuple


def first_pair(xs: List[Tuple[int, int]]) -> Tuple[int, int]:
    return xs[0]


def unpack_param(xs: List[Tuple[int, int]]) -> int:
    a, b = xs[0]
    return a + b


def main() -> None:
    h: List[Tuple[int, int]] = [(1, 2)]

    # Cross-function tuple return, unpacked at the call site.
    a, b = first_pair(h)
    assert a == 1 and b == 2

    # Subscript-and-unpack of a list[Tuple] parameter.
    assert unpack_param(h) == 3

    # Three-component tuple element.
    g: List[Tuple[int, int, int]] = [(4, 5, 6)]
    x, y, z = g[0]
    assert x == 4 and y == 5 and z == 6

    # Lowercase tuple annotation behaves identically.
    k: list[tuple[int, int]] = [(7, 8)]
    p, q = k[0]
    assert p == 7 and q == 8

    # Single-element tuple annotation (no elts list) resolves to a 1-component
    # struct, so a direct chained read works.
    s: List[Tuple[int]] = [(9,)]
    assert s[0][0] == 9


main()
