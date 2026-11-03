# GitHub #9053: the element is an int, so it is not a float.
def g(b):
    dp = [0, 0, 0]
    return dp[b]


a = g(2)
assert isinstance(a, float)
