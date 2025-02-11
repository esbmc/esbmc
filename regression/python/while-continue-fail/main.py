count : int = 0
i:int = 0

while i < 5:
    if (i > 0):
        i = i + 1
        continue

    count = count + 1
    i = i + 1

assert count == 0  # count == 1
