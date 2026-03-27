# Recursive generator with nondet float elements.
# Exercises the bytewise fallback path in __ESBMC_list_eq when
# type_ids differ (void*/Any vs floatbv) but sizes match (both 8 bytes).
def f(arr):
    for x in arr:
        for y in f([]):
            yield y
        yield x

a = nondet_float()
b = nondet_float()
assert list(f([a, b])) == [a, b]
