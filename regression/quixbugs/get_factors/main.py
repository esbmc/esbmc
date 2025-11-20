
def get_factors(n):
    if n == 1:
        return []

    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return [i] + get_factors(n // i)

    return [n]

"""
def get_factors(n):
    if n == 1:
        return []

    for i in range(2, n + 1):
        if n % i == 0:
            return [i] + get_factors(n // i)

    return []
"""

assert get_factors([1]) == []
assert get_factors([100]) == [2, 2, 5, 5]
assert get_factors([101]) == [101]
assert get_factors([104]) == [2, 2, 2, 13]
assert get_factors([2]) == [2]
assert get_factors([3]) == [3]
assert get_factors([17]) == [17]
assert get_factors([63]) == [3, 3, 7]
assert get_factors([74]) == [2, 37]
assert get_factors([73]) == [73]
assert get_factors([9837]) == [3, 3, 1093]
