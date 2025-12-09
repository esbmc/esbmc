count = 0
last = 0
for y in range(2, 10, 3):  # 2, 5, 8
    count += 1
    last = y
assert count == 3 and last == 8
