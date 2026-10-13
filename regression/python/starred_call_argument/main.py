# `f(*args)` on a parameter that is passed a list or tuple is spelled out as
# positional arguments, with its length checked where the call is made.
def add(a, b):
    return a + b


def add3(a, b, c):
    return a + b + c


def check(args, expected):
    assert add(*args) == expected


def check_prefix(rest, expected):
    assert add3(1, *rest) == expected


def check_tuple(pair, expected):
    assert add(*pair) == expected


def main():
    check([1, 2], 3)
    check([5, 5], 10)
    check_prefix([2, 3], 6)
    check_tuple((4, 6), 10)


main()
