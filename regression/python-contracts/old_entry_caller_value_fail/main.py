# The control for the snapshot: __ESBMC_old must hold the arbitrary value the
# parameter was given on entry, not the argument main passes. `old(n) == 3` is
# true of the actual call and false of an arbitrary n, so it verifies in the
# caller-dependent tier and must fail here.
#
# Without a control of this shape the entry-harness tests cannot be told from
# ones where the parameter was never havoc'd -- old_entry passes either way.
def bump(n: int) -> int:
    __ESBMC_requires(n > 0 and n < 1000)
    __ESBMC_ensures(__ESBMC_old(n) == 3)
    n = n + 1
    return n

def main() -> None:
    y: int = bump(3)
    assert y == 4

main()
