# Recursive generator: verify that genuinely unequal lists still fail.
# __ESBMC_assume(a != b) ensures the assertion is false, so the fix must
# not suppress this true failure via the bytewise fallback.
def f(arr):
    for x in arr:
        for y in f([]):
            yield y
        yield x


a: int
b: int
__ESBMC_assume(a != b)
assert list(f([a])) == [b]
