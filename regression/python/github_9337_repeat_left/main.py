# The count comes first: n * [x].
def pick(n):
    dp = (n + 1) * ["a"]
    return dp[n]


r = pick(2)
assert isinstance(r, str)
assert r == "a"
