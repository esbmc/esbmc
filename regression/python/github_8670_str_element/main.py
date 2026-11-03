# A list parameter holding a str beside an int: a constant index reads each
# element at its own type, so the str can be sliced (#8670).
def head(d):
    s = d[0]
    return s[:2]


def size(d):
    return len(d[0]) + d[1]


assert head(["abcd", 2]) == "ab"
assert head(["xyzw", 5]) == "xy"
assert size(["abcd", 2]) == 6
