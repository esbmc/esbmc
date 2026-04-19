
def powerset(arr):
    if arr:
        first, *rest = arr
        rest_subsets = powerset(rest)
        return rest_subsets + [[first] + subset for subset in rest_subsets]
    else:
        return [[]]

"""
def powerset(arr):
    if arr:
        first, *rest = arr
        rest_subsets = powerset(rest)
        return [[first] + subset for subset in rest_subsets] + rest_subsets
    else:
        return [[]]
"""

assert powerset(["a", "b", "c"]) == [
    [],
    ["c"],
    ["b"],
    ["b", "c"],
    ["a"],
    ["a", "c"],
    ["a", "b"],
    ["a", "b", "c"],
]

assert powerset(["a", "b"]) == [
    [],
    ["b"],
    ["a"],
    ["a", "b"],
]

assert powerset(["a"]) == [
    [],
    ["a"],
]

assert powerset([]) == [
    [],
]

assert powerset(["x", "df", "z", "m"]) == [
    [],
    ["m"],
    ["z"],
    ["z", "m"],
    ["df"],
    ["df", "m"],
    ["df", "z"],
    ["df", "z", "m"],
    ["x"],
    ["x", "m"],
    ["x", "z"],
    ["x", "z", "m"],
    ["x", "df"],
    ["x", "df", "m"],
    ["x", "df", "z"],
    ["x", "df", "z", "m"],
]
