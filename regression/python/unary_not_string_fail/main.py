# Negative variant of unary_not_string: `not "x"` is False (a non-empty string
# is truthy), so asserting it equals True must make verification FAIL. Guards
# against the IREP2 `not`-on-string lowering collapsing to a constant.
def is_empty(s: str) -> bool:
    return not s


assert is_empty("x") == True
