g: int = 0

def pure(n: int) -> int:
    __ESBMC_requires(n > 0)
    __ESBMC_assigns()
    __ESBMC_ensures(__ESBMC_return_value == n)
    return n

def main() -> None:
    global g
    g = 5
    v: int = pure(3)
    assert g == 5 and v == 3

main()
