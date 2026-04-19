i: int = 0
count: int = 0

while i < 5:
    i = i + 1
    if i == 3:
        continue
    count = count + 1

assert count == 4  # i=3 was skipped
