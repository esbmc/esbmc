def safe_max(xs: list[int]) -> int:
    if xs:
        return max(xs)
    return 0

assert safe_max([]) == 0
assert safe_max([3, 1, 2]) == 3
