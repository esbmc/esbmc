# Failing variant of issue #3811: ensures the conversion-time fallback does
# not mask genuine assertion violations downstream.


def f(items):
    w, v = items[0]
    return int(w) + int(v)


assert f([[1, 2]]) == 99
