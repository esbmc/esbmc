def validate_category(category: str) -> None:
    i = 0
    while i < len(category):
        assert category[i].isalpha()
        i += 1


def main() -> None:
    category = "Livros"
    validate_category(category)


main()
