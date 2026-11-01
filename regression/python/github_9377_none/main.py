# #9377: a str parameter holding None is a false and/or operand.
def pick(s: str) -> str:
    return s or "d"


assert pick(None) == "d"
assert pick("") == "d"
assert pick("q") == "q"
