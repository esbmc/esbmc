items = [[2, 3], [4, 5, 6]]
k = 0
for i in range(2):
    try:
        w, v = items[i]
    except ValueError:
        k = k + 1
assert k == 1
