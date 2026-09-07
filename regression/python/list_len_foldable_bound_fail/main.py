# len() returns the list's size directly, so a loop bounded by it unwinds the
# list's own length rather than --unwind. sum(), max(), min() and the slice
# lowering are all bounded that way.
xs = [3, 1, 4, 1, 5]

assert len(xs) == 5
assert sum(xs) == 88
assert max(xs) == 5
assert min(xs) == 1
assert len(xs[1:]) == 4

i: int = 0
total: int = 0
while i < len(xs):
    total = total + xs[i]
    i = i + 1
assert total == 14

xs.append(9)
assert len(xs) == 6
xs.pop()
assert len(xs) == 5
