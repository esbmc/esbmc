# Basic starred unpacking from list variable
def f(arr):
    first, *rest = arr
    return rest

assert f([1, 2]) == [2]
assert f([1, 2, 3]) == [2, 3]
