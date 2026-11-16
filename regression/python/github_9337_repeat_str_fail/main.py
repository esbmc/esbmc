# The elements are strs, so asking for an int is false.
def pick(n):
    dp = ["a"] * (n + 1)
    return dp[n]


r = pick(2)
assert isinstance(r, int)
