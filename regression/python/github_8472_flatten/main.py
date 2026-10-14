# Recursive flatten over a list parameter whose elements are lists and ints:
# each element's isinstance() must follow its own type (#8472).
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
