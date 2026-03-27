def f(arr: list[int]) -> list[int]:
    return arr[1:]


l: list[int] = f([1, 2])
assert l[0] == 2
