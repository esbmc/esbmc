def parse(s: str) -> int:
    try:
        return int(s)
    except ValueError:
        return -1

assert parse("abc") == -1
