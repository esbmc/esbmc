def my_max(x:list):
    i = 1
    m = x[0]

    while i < len(x):
        if x[i] > m:
            m = x[i]
        i = i + 1

    return m


l = [1, 2, 3]
assert my_max(l) == 3
