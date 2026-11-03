def gen(n: int):
    if n > 0:
        yield n
        yield from gen(n - 1)


total = 0
for x in gen(2):
    total += x
assert total == 3
