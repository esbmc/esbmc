# An unsupported intrinsic outside any clause is reported as the unsupported
# call it is, rather than blamed on the contract clauses above it.
def bump(x: int) -> int:
    __ESBMC_requires(x > 0)
    __ESBMC_ensures(__ESBMC_return_value > x)
    __ESBMC_is_fresh(x, 8)
    return x + 1

def main() -> None:
    y: int = bump(5)
    assert y == 6

main()
