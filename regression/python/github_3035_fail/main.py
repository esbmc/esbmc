MAX_LEN = 20


def validate_price(price: str) -> None:
    assert price.strip() != "", "O preço não pode conter apenas espaços."


def main() -> None:
    price = "    "
    validate_price(price)


main()
