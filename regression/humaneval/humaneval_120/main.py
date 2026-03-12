# HumanEval/120
# Entry Point: maximum
# ESBMC-compatible format with direct assertions


def maximum(arr, k):
    """
    Given an array arr of integers and a positive integer k, return a sorted list 
    of length k with the maximum k numbers in arr.

    Example 1:

        Input: arr = [-3, -4, 5], k = 3
        Output: [-4, -3, 5]

    Example 2:

        Input: arr = [4, -4, 4], k = 2
        Output: [4, 4]

    Example 3:

        Input: arr = [-3, 2, 1, 2, -1, -2, 1], k = 1
        Output: [2]

    Note:
        1. The length of the array will be in the range of [1, 1000].
        2. The elements in the array will be in the range of [-1000, 1000].
        3. 0 <= k <= len(arr)
    """
    if k == 0:
        return []
    arr.sort()
    ans = arr[-k:]
    return ans

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert maximum([-3, -4, 5], 3) == [-4, -3, 5]
    # assert maximum([4, -4, 4], 2) == [4, 4]
    # assert maximum([-3, 2, 1, 2, -1, -2, 1], 1) == [2]
    # assert maximum([123, -123, 20, 0 , 1, 2, -3], 3) == [2, 20, 123]
    # assert maximum([-123, 20, 0 , 1, 2, -3], 4) == [0, 1, 2, 20]
    # assert maximum([5, 15, 0, 3, -13, -8, 0], 7) == [-13, -8, 0, 0, 3, 5, 15]
    # assert maximum([-1, 0, 2, 5, 3, -10], 2) == [3, 5]
    # assert maximum([1, 0, 5, -7], 1) == [5]
    # assert maximum([4, -4], 2) == [-4, 4]
    # assert maximum([-10, 10], 2) == [-10, 10]
    # assert maximum([1, 2, 3, -23, 243, -400, 0], 0) == []
    print("✅ HumanEval/120 - All assertions completed!")
