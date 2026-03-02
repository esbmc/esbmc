caracteres_especiais_invalidos = '!@#$%^&*()_+=-[]{}|;:,.<>?/~`'


def validate_category(category: str):
    for char in category:
        assert char not in caracteres_especiais_invalidos, "A categoria não deve conter caracteres especiais."


def main() -> None:
    category = "academia"
    validate_category(category)


main()
