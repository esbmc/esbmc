import re


def main() -> None:
    m = re.match(r"(\d+)", "123abc")
    assert m is not None
    assert m.group(1) == "123"


main()
