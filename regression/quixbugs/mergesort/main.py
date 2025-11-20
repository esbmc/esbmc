
def mergesort(arr):
    def merge(left, right):
        result = []
        i = 0
        j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:] or right[j:])
        return result

    if len(arr) <= 1:
        return arr
    else:
        middle = len(arr) // 2
        left = mergesort(arr[:middle])
        right = mergesort(arr[middle:])
        return merge(left, right)

assert mergesort([]) == []
assert mergesort([1, 2, 6, 72, 7, 33, 4]) == [1, 2, 4, 6, 7, 33, 72]
assert mergesort([3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3]) == [1, 1, 2, 3, 3, 3, 4, 5, 5, 5, 6, 7, 8, 9, 9, 9]
assert mergesort([5, 4, 3, 2, 1]) == [1, 2, 3, 4, 5]
assert mergesort([5, 4, 3, 1, 2]) == [1, 2, 3, 4, 5]
assert mergesort([8, 1, 14, 9, 15, 5, 4, 3, 7, 17, 11, 18, 2, 12, 16, 13, 6, 10]) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
assert mergesort([9, 4, 5, 2, 17, 14, 10, 6, 15, 8, 12, 13, 16, 3, 1, 7, 11]) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
assert mergesort([13, 14, 7, 16, 9, 5, 24, 21, 19, 17, 12, 10, 1, 15, 23, 25, 11, 3, 2, 6, 22, 8, 20, 4, 18]) == list(range(1, 26))
assert mergesort([8, 5, 15, 7, 9, 14, 11, 12, 10, 6, 2, 4, 13, 1, 3]) == list(range(1, 16))
assert mergesort([4, 3, 7, 6, 5, 2, 1]) == [1, 2, 3, 4, 5, 6, 7]
assert mergesort([4, 3, 1, 5, 2]) == [1, 2, 3, 4, 5]
assert mergesort([5, 4, 2, 3, 6, 7, 1]) == [1, 2, 3, 4, 5, 6, 7]
assert mergesort([10, 16, 6, 1, 14, 19, 15, 2, 9, 4, 18, 17, 12, 3, 11, 8, 13, 5, 7]) == list(range(1, 20))
assert mergesort([10, 16, 6, 1, 14, 19, 15, 2, 9, 4, 18]) == [1, 2, 4, 6, 9, 10, 14, 15, 16, 18, 19]

