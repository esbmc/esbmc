def minimumCoins_v3(prices: list[int]) -> int:

    def dfs(i, free_until):
        if i >= len(prices):
            return 0

        res: int = prices[i] + dfs(i + 1, min(len(prices) - 1, i + i + 1))

        if free_until >= i:
            res = min(res, dfs(i + 1, free_until))

        return res

    return dfs(0, -1)


assert minimumCoins_v3([1, 2, 3]) == 2
