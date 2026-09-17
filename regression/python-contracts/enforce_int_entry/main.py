# Entry-harness tier: --function makes the enforced function the entry point, so
# every parameter is an arbitrary value the requires clause constrains, and the
# claim is caller-independent. Without it ESBMC follows the call chain from
# main and only ever sees the argument written there.
#
# The bound is what makes the postcondition true rather than merely true of the
# caller's argument: 2 * x overflows a 64-bit int, so `2 * x > x` does not hold
# for every x > 0 (enforce_int_entry_unbounded_fail).
def double(x: int) -> int:
    __ESBMC_requires(x > 0 and x < 1000000)
    __ESBMC_ensures(__ESBMC_return_value > x)
    return 2 * x

def main() -> None:
    y: int = double(5)
    assert y == 10

main()
