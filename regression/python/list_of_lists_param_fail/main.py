# An inner list read out of a list of lists used to be rebuilt from the
# bytes of its pointer, so its size was unknown and the loop over it never
# stopped unwinding.
def total(arr: list[int]) -> int:
    s = 0
    for x in arr:
        s += x
    return s


rows = [[3, 1, 2]]
assert total(rows[0]) == 5
