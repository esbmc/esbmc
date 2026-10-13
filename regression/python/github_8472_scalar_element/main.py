# A parameter filled with the int element of a mixed list parameter takes the
# int type, as `f(*d)` spells it out (#8472).
def pick(arr, k):
    return arr[k]


def check(d, expected):
    assert pick(d[0], d[1]) == expected


check([[3, 4, 5], 1], 4)
check([[7, 8], 0], 7)
