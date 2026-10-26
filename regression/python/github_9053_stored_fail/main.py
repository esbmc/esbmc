# GitHub #9053: a store into the list changes the element type it was built
# with, so the literal does not type the element read.
def g(b):
    dp = [0, 0, 0]
    dp[1] = 1.5
    return dp[b]


a = g(1)
assert isinstance(a, int)
