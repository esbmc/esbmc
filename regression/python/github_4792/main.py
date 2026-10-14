# Tuple unpacking with subscript/attribute targets (the swap idiom) used to
# crash the frontend: "Tuple unpacking only supports simple names, not
# Subscript". The store now routes through the normal single-assignment path
# and the literal const-fold is invalidated for tuple-target subscript
# mutations, so reads see the swap (#4792).
a = [1, 2]
a[0], a[1] = a[1], a[0]
assert a[0] == 2 and a[1] == 1

b = [10, 20, 30]
i = 0
k = 2
b[i], b[k] = b[k], b[i]
assert b[0] == 30 and b[2] == 10


def swap(p, x, y):
    p[x], p[y] = p[y], p[x]
    return p


r = swap([1, 2, 3], 0, 2)
assert r[0] == 3 and r[1] == 2 and r[2] == 1

# Three-way rotation through subscript targets.
c = [1, 2, 3]
c[0], c[1], c[2] = c[2], c[0], c[1]
assert c[0] == 3 and c[1] == 1 and c[2] == 2
