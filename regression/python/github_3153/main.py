import re

s: str = "foo"
match: re.Match[str] | None = re.match(r"foo", s)
if match is not None:
    pass
