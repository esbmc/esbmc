from random import randint

cond: int = randint(0, 1)
counter: int = 0

while counter < 2:
    if cond == 1:
        break
    # counter not updated if cond == 1
    counter = counter + 1

assert counter == 2  # FAILS if cond == 1
