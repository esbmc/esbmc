numbers = [1, 2]
total = 0

for i, x in enumerate(numbers):
    for j, y in enumerate(numbers):
        total += (i + 1) * (j + 1) * y

assert total == 15