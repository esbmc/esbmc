def validate_no_letters(price: str) -> None:
    assert all(not ('a' <= c.lower() <= 'z') for c in price), "Erro: O valor contém letras."

def main() -> None:
    price = "123a"
    validate_no_letters(price)

main()
