def head(d):
    s = d[0]
    return s[:2]


assert head(["abcd", 2]) == "ab"
assert head(["xyzw", 5]) == "xz"
