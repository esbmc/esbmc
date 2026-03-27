MAX_LEN = 20
caracteres_especiais_invalidos = '!@#$%^&*()_+=-[]{}|;:,.<>?/~`'

def validate_title(title: str):
    for char in title:
        char_encontrado = False
        for caractere_especial in caracteres_especiais_invalidos:
            if char == caractere_especial:
                char_encontrado = True
        assert not char_encontrado, "O título não deve conter caracteres especiais."

def main() -> None:
    title = "Livro$"
    validate_title(title)

main()
