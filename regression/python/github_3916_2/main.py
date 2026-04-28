def first_char_or_empty(s: str) -> str:
    if s:
        return s[0]
    return ""


assert first_char_or_empty("hello") == "h"
assert first_char_or_empty("") == ""
