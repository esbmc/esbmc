def f(x: int) -> int:
    return 1


i = 0
while 0 < i + f(i) < 3:
    i += 1
assert i == 0
