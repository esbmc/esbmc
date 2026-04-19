nums = [1, 2, 3, 4, 5]
total = 0
for n in nums:
    if n % 2 == 0:
        continue
    total += n
assert total == 9  # 1 + 3 + 5
