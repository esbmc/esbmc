g: int = 0

def pure(n: int) -> int:
    global g
    __ESBMC_requires(n > 0)
    __ESBMC_assigns()
    __ESBMC_ensures(__ESBMC_return_value == n)
    g = n
    return n

def main() -> None:
    v: int = pure(3)

main()
