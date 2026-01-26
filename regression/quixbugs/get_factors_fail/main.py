
def get_factors(n):
    if n == 1:
        return []

    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return [i] + get_factors(n // i)

    return [n]

assert get_factors(1) == []
assert get_factors(17) == [17]
assert get_factors(73) == [74] #should fail
