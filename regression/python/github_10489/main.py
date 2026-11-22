def f(arr: list[int]) -> int:
    ends = {}
    n = 0
    for i in range(len(arr)):
        if i > 0 and arr[ends[1]] < 5:
            n = n + 1
        ends[1] = i
    return n


assert f([3, 7]) == 1
