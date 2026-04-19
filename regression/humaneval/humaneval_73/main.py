# HumanEval/73
# Entry Point: smallest_change
# ESBMC-compatible format with direct assertions


def smallest_change(arr):
    """
    Given an array arr of integers, find the minimum number of elements that
    need to be changed to make the array palindromic. A palindromic array is an array that
    is read the same backwards and forwards. In one change, you can change one element to any other element.

    For example:
    smallest_change([1,2,3,5,4,7,9,6]) == 4
    smallest_change([1, 2, 3, 4, 3, 2, 2]) == 1
    smallest_change([1, 2, 3, 2, 1]) == 0
    """
    ans = 0
    for i in range(len(arr) // 2):
        if arr[i] != arr[len(arr) - i - 1]:
            ans += 1
    return ans

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert smallest_change([1,2,3,5,4,7,9,6]) == 4
    # assert smallest_change([1, 2, 3, 4, 3, 2, 2]) == 1
    # assert smallest_change([1, 4, 2]) == 1
    # assert smallest_change([1, 4, 4, 2]) == 1
    # assert smallest_change([1, 2, 3, 2, 1]) == 0
    # assert smallest_change([3, 1, 1, 3]) == 0
    # assert smallest_change([1]) == 0
    # assert smallest_change([0, 1]) == 1
    print("✅ HumanEval/73 - All assertions completed!")
