import re

PATTERN = r"ab"


def is_valid(s: str) -> bool:
    return bool(re.match(PATTERN, s))


assert is_valid("aa")
