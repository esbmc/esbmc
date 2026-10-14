import re


def main() -> None:
    m = re.match(r"(\d+)", "42x")
    g = m.groups()
    assert g[0] == "42"
    assert m.group(0) == "42"


main()
