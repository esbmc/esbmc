xs = [1, 5, 6]
n = 0
while [v for v in xs if v > 3]:
    xs.pop()
    n += 1
else:
    n += 10
# Only one element above 3 is popped before the else clause in this
# variant's assertion, which CPython rejects.
assert n == 11
