total = 0
for i in range(5):
    total = total + i
assert total == 10

count = 0
for i in range(2, 10, 3):
    count = count + 1
assert count == 3

seen = 0
for i in range(3):
    if i == 1:
        continue
    if i == 2:
        break
    seen = seen + 1
assert seen == 1
