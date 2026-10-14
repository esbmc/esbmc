# Regression test for issue #3811: list-variable tuple unpacking with an
# untyped function parameter (e.g. `w, v = items[i - 1]`) used to abort the
# conversion with `Cannot determine element type for list variable unpacking`.
# The frontend now falls back to `any_type()` when no annotation or
# list_type_map entry is available, matching how plain subscript reads degrade.


def f(items):
    w, v = items[0]
    return int(w) + int(v)


def g(items):
    total = 0
    for i in range(1, len(items) + 1):
        w, v = items[i - 1]
        total += int(w) + int(v)
    return total


assert f([[1, 2]]) == 3
assert g([[1, 2], [3, 4]]) == 10
