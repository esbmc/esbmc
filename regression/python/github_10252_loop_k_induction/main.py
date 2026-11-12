def halve_all(n: int) -> int:
    total = 0
    for i in range(n):
        total = total + i // 2
    return total


print(halve_all(10))
