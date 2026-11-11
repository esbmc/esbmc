# Issue #9388: str.split("") raises ValueError: empty separator.
s: str = "a b"
parts: list[str] = s.split("")
