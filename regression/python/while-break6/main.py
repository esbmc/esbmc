from random import randint

skip: int = randint(0, 4)
i: int = 0
count: int = 0

while i < 5:
    i = i + 1
    if i == skip:
        continue
    count = count + 1

assert count == 4 or count == 5
