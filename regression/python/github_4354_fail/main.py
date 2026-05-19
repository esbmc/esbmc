import re


def main() -> None:
    m = re.match(r"(\d+)", "123abc")
    assert m is not None
    # group(1) is "123", not "456" — must FAIL.
    assert m.group(1) == "456"


main()
