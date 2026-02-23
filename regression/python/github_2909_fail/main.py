MAX_LEN = 20

def validate_title(title: str)-> None:
    if title.lstrip() != title:
        raise AssertionError("O título não deve começar com espaço.")

def main() -> None:
    title = " Livro"
    validate_title(title)

main()
