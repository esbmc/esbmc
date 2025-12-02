
def validate_code(code: str) -> None:
    repeats = code.count("VIP")
    assert repeats <= 1, "O código promocional não pode repetir o prefixo"


def main() -> None:
    code = "PROMO-VIP"
    validate_code(code)


main()
