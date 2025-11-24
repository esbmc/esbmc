def validate_category(category: str) -> None:
    for char in category:
        assert char.isalpha()

def main() -> None:
    category = "Livros"
    validate_category(category)

main()
