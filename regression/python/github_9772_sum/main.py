def f(n):
    r = range(n)
    total = 0
    for i in r:
        total += i
    return total


assert f(4) == 6
