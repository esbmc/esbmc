def validate_price(price: str) -> None:
    if "." in price:
        decimals = len(price.split(".")[1])
    else:
        decimals = 0
    assert decimals <= 2, "O preço não pode ter mais de duas casas decimais."


def main() -> None:
    price = "12.345"
    validate_price(price)


main()
