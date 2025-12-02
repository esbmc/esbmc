
def validate_price(price: str) -> str:
    cleaned = price.strip()
    assert cleaned == "24.50"
    return cleaned


def main() -> None:
    price = "\n\t  24.50 \r\n"
    validate_price(price)


main()
