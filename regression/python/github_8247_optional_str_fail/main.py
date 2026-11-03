from typing import Optional


def up(name: Optional[str]) -> str:
    return name.upper()


def main() -> None:
    assert up("ab") == "AB"
    up(None)


main()
