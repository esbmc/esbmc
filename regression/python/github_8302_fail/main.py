# `expected` gets `[]` at one call site and `[2]` at the other, and must still
# be compared by content.
def f(n):
    xs = []
    if n > 1:
        xs.append(n)
    return xs


def check(n, expected):
    assert f(n) == expected


def main():
    check(1, [])
    check(2, [3])


main()
