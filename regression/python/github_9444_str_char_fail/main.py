# Issue #9444: a str element is not an int; CPython raises TypeError here, so
# round() must not return it unchanged.
s: str = "ab"
r = round(s[0], 1)
