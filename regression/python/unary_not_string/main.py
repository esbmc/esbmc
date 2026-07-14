# Pins the `not <str>` truthiness contract migrated to IREP2 in Phase 4.4
# (converter_unop.cpp): an empty string is falsy, so `not ""` is True; any
# non-empty string is truthy, so `not s` is False.
def is_empty(s: str) -> bool:
    return not s


assert is_empty("") == True
assert is_empty("x") == False
assert is_empty("hello") == False
