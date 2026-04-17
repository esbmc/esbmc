MAX_LEN = 20

def validate_title(title: str)-> None:
    if title.rstrip() != title:
        raise AssertionError("O título não deve terminar com espaço.")

def main() -> None:
    title = "Livro "
    validate_title(title)

main()

