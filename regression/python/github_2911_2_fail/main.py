MAX_LEN = 20
def validate_title_not_only_spaces(title: str)-> None:
    only_spaces = True
    for char in title:
        if not char.isspace():
            only_spaces = False
    assert not only_spaces, "O título não pode ser composto apenas por espaços."

def main() -> None:
    title = "   "  
    validate_title_not_only_spaces(title)

main()
