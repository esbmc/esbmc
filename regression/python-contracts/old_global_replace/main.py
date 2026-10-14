# The replace side havocs the module global, so the ensures constrains a fresh
# value rather than contradicting the caller's. Without that the ASSUME is
# assume-false and the assertion below is never reached.
g: int = 7

def touch(k: int) -> int:
    __ESBMC_requires(k > 0)
    __ESBMC_ensures(g == __ESBMC_old(g) + 1)
    return k

def main() -> None:
    v: int = touch(3)
    assert False

main()
