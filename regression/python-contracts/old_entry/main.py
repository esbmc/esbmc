# __ESBMC_old under the entry harness: the snapshot must hold the arbitrary
# value the parameter was given on entry, not the argument main passes.
def bump(n: int) -> int:
    __ESBMC_requires(n > 0 and n < 1000)
    __ESBMC_ensures(__ESBMC_return_value == __ESBMC_old(n) + 1)
    n = n + 1
    return n

def main() -> None:
    y: int = bump(3)
    assert y == 4

main()
