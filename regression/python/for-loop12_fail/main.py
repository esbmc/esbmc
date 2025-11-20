word = [10, 20, 30, 40]
found = False
for item in word:
    if item == 50:  # 50 is not in the list
        found = True
assert found  # This will fail
