# The three sibling shapes of github_4715_irep2_native_body_ternary_loc_01:
# the same unlocated `if2t` correction term reaching the native RETURN, OTHER
# and FUNCTION_CALL arms instead of the ASSIGN one (W1-loc, esbmc/esbmc#4715).
def g(v: int) -> int:
    return v


def in_return(n: int, x: int) -> int:
    return (x + n // x) // 2


def in_statement(n: int, x: int) -> None:
    (x + n // x) // 2


def in_argument(n: int, x: int) -> None:
    g((x + n // x) // 2)


assert in_return(9, 5) == 3
in_statement(9, 5)
in_argument(9, 5)
