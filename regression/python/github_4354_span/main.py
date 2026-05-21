import re


def main() -> None:
    m = re.match(r"(\d+)", "42x")
    s = m.span(0)
    assert s[0] == 0
    assert s[1] == 2


main()
