a = [1, 2]

def f(n: int) -> int:
    __ESBMC_ensures(a[5] == 1)
    return n

def main() -> None:
    v: int = f(1)
    assert v == 1

main()
