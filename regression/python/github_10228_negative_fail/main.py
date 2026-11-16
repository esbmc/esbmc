# Issue #10228: 2 ** n is a float for a negative n, which range() rejects
def width(n: int) -> int:
    count = 0
    for mask in range(2 ** n):
        count += 1
    return count


width(-1)
