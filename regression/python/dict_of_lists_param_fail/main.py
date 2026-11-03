# A list stored as a dict value keeps naming its list.
def total(arr: list[int]) -> int:
    s = 0
    for x in arr:
        s += x
    return s


graph = {1: [2, 3]}
assert total(graph[1]) == 4
