def find_in_sorted(arr, x):
    def binsearch(start, end):
        if start == end:
            return -1
        mid = start + (end - start) // 2
        if x < arr[mid]:
            return binsearch(start, mid)
        elif x > arr[mid]:
            return binsearch(mid + 1, end)
        else:
            return mid

    return binsearch(0, len(arr))
    
assert find_in_sorted([3, 4, 5, 5, 5, 5, 6], 5) == 3
assert find_in_sorted([1, 2, 3, 4, 6, 7, 8], 5) == -1
# assert find_in_sorted([1, 2, 3, 4, 6, 7, 8], 4) == 3
# assert find_in_sorted([2, 4, 6, 8, 10, 12, 14, 16, 18, 20], 18) == 8
# assert find_in_sorted([3, 5, 6, 7, 8, 9, 12, 13, 14, 24, 26, 27], 0) == -1
# assert find_in_sorted([3, 5, 6, 7, 8, 9, 12, 12, 14, 24, 26, 27], 12) == 6
# assert find_in_sorted([24, 26, 28, 50, 59], 101) == -1
