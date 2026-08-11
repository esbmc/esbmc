def neg(b: bool) -> bool:
    __ESBMC_requires(b or not b)
    __ESBMC_ensures(__ESBMC_return_value != b)
    return not b

def main() -> None:
    c: bool = neg(True)
    assert c == False

main()
