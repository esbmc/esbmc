# A list comprehension used directly as the iterable of another comprehension
# — e.g. list(filter(p, [..])) desugars to [x for x in [..] if p(x)] — must be
# materialised into its own temp list first. Otherwise the inner comprehension
# reaches the C++ converter unlowered ("Unsupported expression ListComp").
# See esbmc/esbmc#4807 (humaneval_108).
def nested(arr):
    return len([x for x in [y for y in arr]])


def filter_comp(arr):
    return len(list(filter(lambda x: x > 0, [i for i in arr])))


assert nested([]) == 0
assert nested([1, 2, 3]) == 3
assert filter_comp([]) == 0
assert filter_comp([1, -2, 3]) == 2
