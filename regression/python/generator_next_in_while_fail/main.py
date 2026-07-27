# Negative variant: with generator state persisting, `i` reaches exactly 5, so
# the (wrong) `i == 4` postcondition must be falsified.
def gen():
    j = 0
    while j < 10:
        j = j + 1
        yield j


b = gen()
i = 0
while i < 5:
    i = next(b)

assert i == 4
