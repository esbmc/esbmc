def validate_no_empty_parts(price: str) -> None:
    parts = "12".split(".")
    i = 0
    while i < len(parts):
        if parts[i] == "":
            raise AssertionError("Erro: Parte vazia detectada antes ou depois do ponto.")
        i += 1


def main() -> None:
    price = "12"
    validate_no_empty_parts(price)
    assert False


main()
