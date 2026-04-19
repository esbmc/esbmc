
def find_first_in_sorted(arr, x):
    lo = 0
    hi = len(arr)

    while lo < hi:
        mid = (lo + hi) // 2

        if x == arr[mid] and (mid == 0 or x != arr[mid - 1]):
            return mid

        elif x <= arr[mid]:
            hi = mid

        else:
            lo = mid + 1

    return -1

assert find_first_in_sorted([3, 4, 5, 5, 5, 5, 6], 5) == 2
assert find_first_in_sorted([3, 4, 5, 5, 5, 5, 6], 7) == -1
assert find_first_in_sorted([3, 4, 5, 5, 5, 5, 6], 2) == -1
assert find_first_in_sorted([3, 6, 7, 9, 9, 10, 14, 27], 14) == 6
assert find_first_in_sorted([0, 1, 6, 8, 13, 14, 67, 128], 80) == -1
assert find_first_in_sorted([0, 1, 6, 8, 13, 14, 67, 128], 67) == 6
assert find_first_in_sorted([0, 1, 6, 8, 13, 14, 67, 128], 128) == 7
