class C:
    k = 0


C.k = 5
C.k += 1
assert C().k == 6
