# reversed() must reverse order; asserting the original order is wrong and must
# fail under both CPython (AssertionError) and ESBMC (VERIFICATION FAILED).
xs = [1, 2, 3]
assert list(reversed(xs)) == [1, 2, 3]
