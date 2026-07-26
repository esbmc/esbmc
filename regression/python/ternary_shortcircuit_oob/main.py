# A Python conditional expression short-circuits: the untaken branch is never
# evaluated. Here a[c] with c == 1 (index 1, out of range for the 1-element
# list) sits in the untaken branch, so no IndexError may be raised.
def f() -> int:
    a = [10]
    s = 0
    for c in range(0, 2):
        x = a[c] if c < 1 else 99
        s = s + x
    return s


assert f() == 109
