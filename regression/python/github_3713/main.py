MAX_PRICE_LEN: int = 6


def main() -> None:
    price: str = input()
    length_price: int = len(price)
    assert 1 <= length_price <= 5


main()
