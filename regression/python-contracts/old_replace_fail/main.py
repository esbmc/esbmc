# The result overwrites the very variable the snapshot names, which is the
# shape the snapshot has to survive: without one the ensures would constrain
# the post-call value against itself.
def bump(x: int) -> int:
    __ESBMC_requires(x > 0)
    __ESBMC_ensures(__ESBMC_return_value == __ESBMC_old(x) + 1)
    return x + 1

def main() -> None:
    n: int = 5
    n = bump(n)
    assert n == 7

main()
