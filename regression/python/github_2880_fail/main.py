caracteres_especiais_invalidos = '!@#$%^&*()_+=-[]{}|;:,.<>?/~`'
def validate_title(title: str):
    last = title[-1]
    assert not last.isdigit()
    assert last not in caracteres_especiais_invalidos
def main() -> None:
    title = "Livro$"
    validate_title(title)
main()
