# `f(*args)` with a list of fixed length is spelled out as positional
# arguments; the frontend has no Starred expression and refused it before.
def add(a, b):
    return a + b


def check(args, expected):
    assert add(*args) == expected


def main():
    check([1, 2], 3)
    check([5, 5], 11)


main()
