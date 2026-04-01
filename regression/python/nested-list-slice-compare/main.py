def outer(arr):
    def merge(left, right):
        i = 0
        j = 0
        if left[i] <= right[j]:
            return 1
        return 0

    return merge(arr[:1], arr[1:])


assert outer([1, 2]) == 1
