
def validate_title(title: str) -> None:
    ESBMC_index: int = 0
    ESBMC_length: int = len(title)
    last_index: int = ESBMC_length - 1

    while ESBMC_index < ESBMC_length:
        if ESBMC_index == last_index:
            assert title[ESBMC_index] != ' '
        ESBMC_index += 1


def main() -> None:
    title = "Book "
    validate_title(title)


main()
