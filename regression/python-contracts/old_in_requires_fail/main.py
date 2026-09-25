# A precondition already speaks about the pre-state, and only the ensures is
# rewritten, so a snapshot here would be asserted over an undefined symbol.
def bump(x: int) -> int:
    __ESBMC_requires(__ESBMC_old(x) > 0)
    __ESBMC_ensures(__ESBMC_return_value == x + 1)
    return x + 1

def main() -> None:
    y: int = bump(5)
    assert y == 6

main()
