def distributeCandies_v6(n: int, limit: int) -> int:
    ans = 0
    for a in range(limit+1):
        for b in range(limit+1):
            for c in range(limit+1):
                if a+b+c == n: ans += 1
    return ans

assert distributeCandies_v6(2, 4) == 5
