# Issue #9076: a call passing a bytes literal must not be folded as a str.
def first(x):
    return x[0]


assert first(b"ab") == 97
