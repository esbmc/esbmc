# Issue #10130: a later str-and-bool operand's IndexError check runs only
# where the operand is evaluated.
t = ""
r = bool(t and t[0] == "a")
assert not r
assert not (t and t[0] == "a")


def first_is_a(s: str) -> bool:
    return not (s and s[0] == "a")


assert first_is_a("")
assert not first_is_a("ab")
