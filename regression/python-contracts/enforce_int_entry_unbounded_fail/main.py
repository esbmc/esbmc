# The control that gives the entry-harness tier its value: without it, a passing
# entry-harness test cannot be told from one where the parameter was never
# havoc'd at all.
#
# `2 * x > x` reads as arithmetic but is false in the machine type, since 2 * x
# overflows a 64-bit int. It holds for the argument main passes, which is why
# the caller-dependent tier (enforce_int) verifies, and fails here where x is
# arbitrary.
def double(x: int) -> int:
    __ESBMC_requires(x > 0)
    __ESBMC_ensures(__ESBMC_return_value > x)
    return 2 * x

def main() -> None:
    y: int = double(5)
    assert y == 10

main()
