# A generator consumed by `next()` inside a while loop must keep its state
# across calls: the init (`j = 0`) has to run once, before the loop, not on
# every iteration. If it reset each pass, next(b) would always yield 1 and the
# loop would never terminate.
def gen():
    j = 0
    while j < 10:
        j = j + 1
        yield j


b = gen()
i = 0
while i < 5:
    i = next(b)

assert i == 5
