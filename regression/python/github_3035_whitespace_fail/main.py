
def validate_price(price: str) -> None:
    assert price.strip() != "", "Somente espaços em branco são inválidos"


def main() -> None:
    price = "\n\t\r"
    validate_price(price)


main()
