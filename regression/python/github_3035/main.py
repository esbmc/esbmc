def validate_price(price: str) -> None:
    assert price.strip() == "esbmc-python"


def main() -> None:
    text = "   esbmc-python   "
    validate_price(text)


main()
