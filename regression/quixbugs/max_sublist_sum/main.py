
def max_sublist_sum(arr):
    max_ending_here = 0
    max_so_far = 0

    for x in arr:
        max_ending_here = max(0, max_ending_here + x)
        max_so_far = max(max_so_far, max_ending_here)

    return max_so_far

"""
def max_sublist_sum(arr):
    max_ending_here = 0
    max_so_far = 0

    for x in arr:
        max_ending_here = max(max_ending_here + x, 0)
        max_so_far = max(max_so_far, max_ending_here)

    return max_so_far

def max_sublist_sum(arr):
    max_ending_here = 0
    max_so_far = 0

    for x in arr:
        max_ending_here = max(x, max_ending_here + x)
        max_so_far = max(max_so_far, max_ending_here)

    return max_so_far


def max_sublist_sum(arr):
    max_ending_here = 0
    max_so_far = 0

    for x in arr:
        max_ending_here = max(max_ending_here + x, x)
        max_so_far = max(max_so_far, max_ending_here)

    return max_so_far

"""

assert max_sublist_sum([4, -5, 2, 1, -1, 3]) == 5
assert max_sublist_sum([0, -1, 2, -1, 3, -1, 0]) == 4
assert max_sublist_sum([3, 4, 5]) == 12
assert max_sublist_sum([4, -2, -8, 5, -2, 7, 7, 2, -6, 5]) == 19
assert max_sublist_sum([-4, -4, -5]) == 0
assert max_sublist_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6
