# As github_9337_repeat, but the value asserted is false.
def pick(n):
    dp = [7] * (n + 1)
    return dp[n]


r = pick(2)
assert isinstance(r, int)
assert r == 8
