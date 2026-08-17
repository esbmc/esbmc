def outer():
    def inner(*rest):
        return len(rest)

    return inner(1, 2, 3)


def closes_over_a_local():
    base = 10

    def helper(*rest):
        return base + len(rest)

    return helper(1, 2, 3)


def keeps_fixed_parameters():
    def helper(x, *rest):
        return x + len(rest)

    return helper(5, 1, 2)


def sibling_of_a_same_named_nested_def():
    def helper(*rest):
        return len(rest) * 100

    return helper(7, 8, 9)


def called_at_two_arities():
    def helper(*rest):
        return len(rest)

    return helper(1) + helper(2, 3, 4)


assert outer() == 3
assert called_at_two_arities() == 4
assert closes_over_a_local() == 13
assert keeps_fixed_parameters() == 7
assert sibling_of_a_same_named_nested_def() == 300
