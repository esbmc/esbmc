# Caller-dependent tier: without --function ESBMC follows the call chain from
# main, so this establishes the contract for the argument written there, not for
# every x the requires clause admits -- 2 * x overflows a 64-bit int. The
# caller-independent claim is enforce_int_entry.
def double(x: int) -> int:
    __ESBMC_requires(x > 0)
    __ESBMC_ensures(__ESBMC_return_value > x)
    return 2 * x

def main() -> None:
    y: int = double(5)
    assert y == 10

main()
