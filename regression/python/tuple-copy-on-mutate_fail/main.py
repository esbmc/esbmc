# tuple(l) snapshots the list: a later mutation of l is NOT visible
# through the tuple, so asserting the mutated value must fail. Guards
# against modelling tuple(list) as a shared reference to the source list,
# which would wrongly verify this program (missed bug).
l = [1, 2]
t = tuple(l)
l[0] = 9
assert t[0] == 9
