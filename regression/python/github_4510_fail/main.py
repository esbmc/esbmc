def f(n: int) -> int:
    h: int = 0
    result: int = 0
    while h < 3:
        x: int
        if 5 < n:
            x = 5
        else:
            x = n
        result = result + x
        h = h + 1
    return result


r = f(10)
assert r == 0
