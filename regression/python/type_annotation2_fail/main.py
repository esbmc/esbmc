def minimumCoins_v2(prices: list[int]) -> int:
    n = len(prices)
    def dp(i):
        if i >= n:
            return 0
        min_cost = float('inf')
        for j in range(i + 1, i + i + 3):
            min_cost = min(min_cost, dp(j))
        return prices[i] + min_cost
    return dp(0)

assert minimumCoins_v2([1, 2, 3]) == 2
