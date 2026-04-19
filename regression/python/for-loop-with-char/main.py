def check_title(title: str) -> None:
    for char in title:
        assert ('a' <= char <= 'z' or 'A' <= char <= 'Z' or '0' <= char <= '9')

def main() -> None:
    title = "academia123"
    check_title(title)

main()
