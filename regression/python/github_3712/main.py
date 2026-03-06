MAX_LEN: int = 31

def validate_start_with_letter(category: str) -> bool:
    assert category[0].isalpha(), "A categoria deve começar com uma letra."
    return True

def main() -> None:
    category: str = input()
    validate_start_with_letter(category)

main()
