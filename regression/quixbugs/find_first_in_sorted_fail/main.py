def find_first_in_sorted(arr, x):
    lo = 0
    hi = len(arr)

    while lo <= hi:
        mid:int = (lo + hi) // 2

        if x == arr[mid] and (mid == 0 or x != arr[mid - 1]):
            return mid

        elif x <= arr[mid]:
            hi:int = mid

        else:
            lo = mid + 1

    return -1

assert find_first_in_sorted([3, 4, 5, 5, 5, 5, 6], 5) == 2
assert find_first_in_sorted([3, 4, 5, 5, 5, 5, 6], 7) == -1