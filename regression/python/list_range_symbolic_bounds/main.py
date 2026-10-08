# list(range(...)) with a non-constant bound took a path that produced a list
# with no elements (two args) or with its size set and no contents (one arg),
# so reading any element raised a spurious IndexError on a correct program.
# #5929 fixed the module-constant case by folding the bound to a literal; a
# bound that is genuinely symbolic, like a parameter, still reached that path.
def one_arg(n: int) -> None:
    c = list(range(n))
    assert len(c) == n
    assert c[0] == 0


def two_args(a: int, b: int) -> None:
    c = list(range(a, b))
    assert len(c) == b - a
    assert c[0] == a


def backwards(a: int, b: int) -> None:
    # Python gives an empty list when the range runs backwards; the count is
    # clamped rather than reaching the fill loop, whose guard raises
    # ValueError on a negative size.
    assert len(list(range(a, b))) == 0


one_arg(9)
two_args(1, 10)
backwards(5, 2)
