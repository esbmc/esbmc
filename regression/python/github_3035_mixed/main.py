
def normalize_title(title: str) -> str:
    cleaned = title.strip()
    assert cleaned == "Special deal"
    assert "  " not in cleaned
    return cleaned


def main() -> None:
    title = "  Special deal  "
    normalize_title(title)


main()
