def f(a, b):
    r = range(a, b)
    count = 0
    for _ in r:
        count += 1
    return count


assert f(2, 7) == 5
