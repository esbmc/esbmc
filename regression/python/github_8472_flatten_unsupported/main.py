# Each element of the list is read at a loop index, where no single type
# answers for a list beside an int: refused rather than misread (#8472).
def flatten(arr):
    out = []
    for x in arr:
        if isinstance(x, list):
            for y in flatten(x):
                out.append(y)
        else:
            out.append(x)
    return out


assert flatten([[1, [2]], 3]) == [1, 2, 3]
