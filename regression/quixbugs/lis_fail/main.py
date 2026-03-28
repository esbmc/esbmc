
def lis(arr):
    ends = {}
    longest = 0

    for i, val in enumerate(arr):

        prefix_lengths = [j for j in range(1, longest + 1) if arr[ends[j]] < val]

        length = max(prefix_lengths) if prefix_lengths else 0

        if length == longest or val < arr[ends[length + 1]]:
            ends[length + 1] = i
            longest = length + 1

    return longest



"""
Longest Increasing Subsequence
longest-increasing-subsequence


Input:
    arr: A sequence of ints

Precondition:
    The ints in arr are unique

Output:
    The length of the longest monotonically increasing subsequence of arr

Example:
    >>> lis([4, 1, 5, 3, 7, 6, 2])
    3
"""

assert lis() == 0
assert lis(3) == 1
assert lis(10, 20, 11, 32, 22, 48, 43) == 4
assert lis(4, 2, 1) == 1
assert lis(5, 1, 3, 4, 7) == 4
assert lis(4, 1) == 1
assert lis(-1, 0, 2) == 3
assert lis(0, 2) == 2
assert lis(4, 1, 5, 3, 7, 6, 2) == 3
assert lis(10, 22, 9, 33, 21, 50, 41, 60, 80) == 6
assert lis(7, 10, 9, 2, 3, 8, 1) == 3
assert lis(9, 11, 2, 13, 7, 15) == 4
