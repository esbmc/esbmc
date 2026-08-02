def gen(arr):
    def helper():
        # references gen() from a nested helper — gen itself is NOT recursive,
        # so it must not be rewritten as a recursive generator.
        return list(gen([]))
    for x in arr:
        yield x


result = list(gen([1, 2, 3]))
assert result[0] == 1
assert result[1] == 2
assert result[2] == 3
