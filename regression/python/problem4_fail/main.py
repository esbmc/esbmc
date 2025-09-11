def minimumCoins_v1(prices: list[int]) -> int:
    n = len(prices)
    if not n: return 0
    elif n == 1: return prices[0]
    #dp = [float("inf")] * n
    dp = [1, 2, 3]
    for j in range(2):
        dp[j] = prices[0]
    for i in range(1, n):
        price = dp[i - 1] + prices[i]
        for j in range(i, min(n, (i + 1) * 2)):
            dp[j] = min(dp[j], price)
    return dp[-1]


assert minimumCoins_v1([1, 2, 3]) == 2
