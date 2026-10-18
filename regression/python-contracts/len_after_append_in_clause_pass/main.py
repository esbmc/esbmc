def grow(l: list) -> int:
    __ESBMC_ensures(len(l) == 4)
    l.append(1)
    return 0

def main() -> None:
    a = [7, 8, 9]
    grow(a)
    assert len(a) == 4

main()
