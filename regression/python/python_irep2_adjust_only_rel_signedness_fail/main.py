def count_parts(text: str) -> int:
    parts = text.split(".")
    i = 0
    seen = 0
    while i < len(parts):
        seen += 1
        i += 1
    return seen


def main() -> None:
    assert count_parts("0.00") == 3


main()
