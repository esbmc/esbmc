# github #8688: int() of a str accepts whitespace only around the digits.
def parse(s: str) -> int:
    return int(s)


assert parse("  12  ") == 12
assert parse("\t-12\n") == -12
assert parse("1_2 ") == 12
assert parse("+7") == 7
