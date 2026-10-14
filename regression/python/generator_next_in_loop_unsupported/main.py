# Each iteration consumes a different yield, which one inlined copy cannot
# follow.
def gen():
    yield 1
    yield 2


g = gen()
s = 0
i = 0
while i < 2:
    x = next(g)
    s = s + x
    i = i + 1
assert s == 3
