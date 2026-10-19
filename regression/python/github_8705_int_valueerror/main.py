def parse(s: str) -> int:
    try:
        return int(s)
    except ValueError:
        return -1


assert parse("abc") == -1
assert parse("42") == 42
assert parse(" -7 ") == -7
assert parse("1 2") == -1
assert parse("") == -1
