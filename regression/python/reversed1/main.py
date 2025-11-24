def minimumCoins_v6(prices: list[int]) -> int:
    n = len(prices)
    dp = [math.inf] * (n + 1)
    dp[-1] = 0
    for i in reversed(range(n)):
        dp[i] = prices[i] + min(dp[j] for j in range(i + 1, min(2 * i + 2, n) + 1))
    return dp[0]
