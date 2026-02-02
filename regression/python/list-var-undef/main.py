def max(x):
    i = 1
    max = x[0]
    while i < len(l):
        if x[i] > max:
            max = x[i]
        i = i + 1
    return max

l = [1, 2, 3]
assert max(l) == 3
