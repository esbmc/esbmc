# `continue` in a `for ... in range(...)` loop previously skipped the counter
# advance (it was emitted after the body, and continue jumps to the while
# condition), so the loop never terminated. The counter is now advanced before
# the body. Covers default/explicit start, a step, and a nested loop.
s = 0
for i in range(5):
    if i % 2:
        continue
    s += i
assert s == 6

t = 0
for i in range(1, 5):
    if i == 3:
        continue
    t += i
assert t == 7

u = 0
for i in range(0, 10, 2):
    if i == 4:
        continue
    u += i
assert u == 16

r = 0
for i in range(3):
    for j in range(3):
        if j == 1:
            continue
        r += 1
assert r == 6
