# GitHub #9088: with a tuple parameter, the result's len() used to abort
# the encoder.
def f(t: tuple[int, int]) -> int:
    x = [1] + t
    return len(x)


f((2, 3))
