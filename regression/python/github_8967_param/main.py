def pairs(a: list[int], b: list[int]) -> int:
    z = zip(a, b)
    n = 0
    for x, y in z:
        n += x + y
    return n


assert pairs([1, 2, 3], [4, 5]) == 12
