def distributeCandies_v5(n: int, limit: int) -> int:
    answer = 0
    for i in range(limit + 1):
        for j in range(limit + 1):
            k = n - i - j
            if 0 <= k <= limit:
                answer += 1
    return answer


assert distributeCandies_v5(2, 3) == 6
