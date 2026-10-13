# A list parameter holding an int and a list of pairs: each position keeps
# its type, and the pairs keep theirs (#8472).
def first_value(d):
    items = d[1]
    pair = items[0]
    return d[0] + pair[1] + len(d)


assert first_value([100, [[60, 10], [50, 8]]]) == 112
assert first_value([40, [[30, 5]]]) == 47
