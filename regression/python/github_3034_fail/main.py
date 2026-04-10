def validate_no_empty_parts(price: str) -> None:
    partes = price.split('.')
    i = 0
    while i < len(partes):
        if partes[i] == '':
            raise AssertionError("Erro: Parte vazia detectada antes ou depois do ponto.")
        i += 1

def main() -> None:
    price = ".12"
    validate_no_empty_parts(price)

main()
