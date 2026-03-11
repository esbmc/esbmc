MAX_PRICE_LEN: int = 6


def validate_positive_value(price: str, length: int) -> bool:
    value: float = 0.0
    factor: float = 1.0
    i: int = 0
    # logic omitted for simplification
    return True


def main() -> None:
    price: str = input()
    length: int = len(price)
    assert 1 <= length <= 5
    validate_positive_value(price, length)


main()
