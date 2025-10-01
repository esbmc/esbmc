def distributeCandies_v4(n: int, limit: int) -> int:
    limit = min(limit, n)
    ans = 0
    for i in range(limit+1):
        if n-i > limit * 2:
            continue
        ans += (min(limit, n-i) - max(0, n-i-limit) + 1)
    return ans

assert  distributeCandies_v4(2, limit = 3) == 6
