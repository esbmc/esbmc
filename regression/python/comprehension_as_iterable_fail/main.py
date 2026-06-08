# Negative variant of comprehension_as_iterable: a comprehension iterating
# another comprehension must be modelled soundly, so a false assertion about
# its length is a genuine violation ESBMC must catch. See esbmc/esbmc#4807.
def filter_comp(arr):
    return len(list(filter(lambda x: x > 0, [i for i in arr])))


assert filter_comp([1, -2, 3]) == 3
