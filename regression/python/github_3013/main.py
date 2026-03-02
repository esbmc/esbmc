import re

# Any string with two lower case letters
PATTERN = r"^[a-z][a-z]$"


def is_valid(s: str) -> bool:
    return bool(re.match(PATTERN, s))


assert is_valid("aa")
