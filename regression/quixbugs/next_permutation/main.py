
def next_permutation(perm):
    for i in range(len(perm) - 2, -1, -1):
        if perm[i] < perm[i + 1]:
            for j in range(len(perm) - 1, i, -1):
                if perm[i] < perm[j]:
                    next_perm = list(perm)
                    next_perm[i], next_perm[j] = perm[j], perm[i]
                    next_perm[i + 1:] = reversed(next_perm[i + 1:])
                    return next_perm

"""
def next_permutation(perm):
    for i in range(len(perm) - 2, -1, -1):
        if perm[i] < perm[i + 1]:
            for j in range(len(perm) - 1, i, -1):
                if perm[j] > perm[i]:
                    next_perm = list(perm)
                    next_perm[i], next_perm[j] = perm[j], perm[i]
                    next_perm[i + 1:] = reversed(next_perm[i + 1:])
                    return next_perm
"""

assert next_permutation([3, 2, 4, 1]) == [3, 4, 1, 2]
assert next_permutation([3, 5, 6, 2, 1]) == [3, 6, 1, 2, 5]
# assert next_permutation([3, 5, 6, 2]) == [3, 6, 2, 5]
# assert next_permutation([4, 5, 1, 7, 9]) == [4, 5, 1, 9, 7]
# assert next_permutation([4, 5, 8, 7, 1]) == [4, 7, 1, 5, 8]
# assert next_permutation([9, 5, 2, 6, 1]) == [9, 5, 6, 1, 2]
# assert next_permutation([44, 5, 1, 7, 9]) == [44, 5, 1, 9, 7]
# assert next_permutation([3, 4, 5]) == [3, 5, 4]

