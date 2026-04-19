def f(arr:list[int]) -> list[int]:
    return arr[1:]

l:list[int] = f([1, 2])
assert len(l) == 1
