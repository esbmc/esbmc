def knapsack(capacity: int, items: list[list[int]]) -> int:
    n = len(items)
    table = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        weight, value = items[i - 1]
        for j in range(capacity + 1):
            table[i][j] = table[i - 1][j]
            if weight <= j and table[i - 1][j - weight] + value > table[i][j]:
                table[i][j] = table[i - 1][j - weight] + value
    return table[n][capacity]


assert knapsack(3, [[1, 1], [2, 3]]) == 5
