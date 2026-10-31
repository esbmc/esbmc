# A list built by repeating a literal keeps its element type when returned.
def pick(n):
    dp = [7] * (n + 1)
    return dp[n]


r = pick(2)
assert isinstance(r, int)
assert r == 7
assert pick(1) == 7
