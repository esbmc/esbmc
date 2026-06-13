# Negative variant: the asserted sum is wrong, so verification must FAIL.
def sum_pairs():
    total = 0
    for u, v in [(1, 2), (3, 4), (5, 6)]:
        total = total + u * 10 + v
    return total


assert sum_pairs() == 999
