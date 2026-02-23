from math import comb

def distributeCandies_v2(n: int, limit: int) -> int:
    res = comb(n + 2, 2)
    if n >= limit + 1:
        res -= 3 * comb(n - limit + 1, 2)
    if n >= 2 * limit + 2:
        res += 3 * comb(n - 2 * limit, 2)
    if n >= 3 * (limit + 1):
        res -= comb(n - 3 * limit - 1, 2)
    return res

assert  distributeCandies_v2(2, 2) == 5
