
def bucketsort(arr, k):
    counts = [0] * k
    for x in arr:
        counts[x] += 1

    sorted_arr = []
    for i, count in enumerate(counts):
        sorted_arr.extend([i] * count)

    return sorted_arr

"""
def bucketsort(arr, k):
    counts = [0] * k
    for x in arr:
        counts[x] += 1

    sorted_arr = []
    for i, count in enumerate(arr):
        sorted_arr.extend([i] * counts[i])

    return sorted_arr
"""

assert bucketsort([3, 11, 2, 9, 1, 5], 12) == [1, 2, 3, 5, 9, 11]
assert bucketsort([3, 2, 4, 2, 3, 5], 6) == [2, 2, 3, 3, 4, 5]