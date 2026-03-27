def distributeCandies_v3(n: int, limit: int) -> int:
    answer = 0
    for c3 in range(limit + 1):
        c1_min = max(0, n - c3 - limit)
        c1_max = min(limit, n - c3)
        answer += max(c1_max - c1_min + 1, 0)
    return answer


assert distributeCandies_v3(2, 3) == 6
