
def first_up(p):
    for i in range(len(p) - 1):
        if p[i] < p[i + 1]:
            return list(p)


r = first_up([2, 1])
assert r[0] == 2
