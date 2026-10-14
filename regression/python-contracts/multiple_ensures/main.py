# Caller-dependent tier: both postconditions are checked, but against the
# argument main passes. `__ESBMC_return_value > x` does not hold for every
# x > 0, since 2 * x overflows a 64-bit int (enforce_int_entry_unbounded_fail).
def twice(x: int) -> int:
    __ESBMC_requires(x > 0)
    __ESBMC_ensures(__ESBMC_return_value > x)
    __ESBMC_ensures(__ESBMC_return_value == 2 * x)
    return 2 * x

def main() -> None:
    y: int = twice(4)
    assert y == 8

main()
