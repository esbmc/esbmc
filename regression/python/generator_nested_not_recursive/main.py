def gen(arr):
    def helper():
        return list(gen([]))
    for x in arr:
        yield x

result = list(gen([1, 2, 3]))
assert result[0] == 1
assert result[1] == 2
assert result[2] == 3
