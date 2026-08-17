# A local does not exist at the point the snapshot would be taken, so there is
# no pre-call value for it to name.
def bump(x: int) -> int:
    t: int = x + 1
    __ESBMC_ensures(__ESBMC_return_value == __ESBMC_old(t))
    return t

def main() -> None:
    y: int = bump(5)
    assert y == 6

main()
