xs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

total = 0
while xs:
    total += xs.pop()

assert total == 45
