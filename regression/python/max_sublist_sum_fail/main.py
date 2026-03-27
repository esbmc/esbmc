def max_sublist_sum(arr:list[int]) -> int:
    max_ending_here = 0
    max_so_far = 0

    for x in arr:
        max_ending_here = max_ending_here + x
        max_so_far = max(max_so_far, max_ending_here)

    return max_so_far

assert max_sublist_sum([4, -5, 2, 1, -1, 3]) == 5
