from random import randint

skip: int = randint(0, 5)
i: int = 0
count: int = 0

while i < 5:
    if i == skip:
        i = i + 1
        continue
    count = count + 1
    i = i + 1

# Skip could be 0, count could be only 4
assert count == 5  # FAILS if skip in range
