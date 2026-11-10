# len() before the module-level def named len is the builtin (#10159).
w = len([1, 2])


def len(v) -> int:
    return 3


assert w == 2
