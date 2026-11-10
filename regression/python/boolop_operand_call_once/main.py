# A call in a value-producing and/or operand is both tested and selected; it
# must still run once (#10160: `xs.append(0) or 2` appended twice).
n = 0


def f() -> int:
    global n
    n = n + 1
    return 0


def g() -> int:
    global n
    n = n + 10
    return 7


x = 0
v = x or f() or g() or 3
assert v == 7
assert n == 11
w = 4 or f()
assert n == 11
xs = [1]
xs.append(0) or 2
assert len(xs) == 2
i = 0
while (f() or 0) != 0:
    i = i + 1
assert n == 12 and i == 0
