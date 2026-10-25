# GitHub #9053: a list element read with a parameter index keeps its type.
def g(b):
    dp = [0, 0, 0]
    return dp[b]


def h(b):
    s = ["x", "yy", "z"]
    return s[b]


a = g(2)
assert isinstance(a, int)
assert isinstance(h(1), str)
assert h(1) == "yy"
