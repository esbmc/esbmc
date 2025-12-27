from typing import List


def binary_search(arr: List[int], key: int) -> int:
    """
    Returns the index of key in arr if present, otherwise -1.
    Precondition: arr is sorted in non-decreasing order.
    """
    left: int = 0
    right: int = len(arr) - 1

    while left <= right:
        mid: int = (left + right) // 2

        if arr[mid] == key:
            return mid
        elif arr[mid] < key:
            left = mid + 1
        else:
            right = mid - 1

    return -1


def is_sorted(arr: List[int]) -> bool:
    i: int = 0
    while i + 1 < len(arr):
        if arr[i] > arr[i + 1]:
            return False
        i += 1
    return True


def main() -> None:
    # Concrete, well-typed test array
    x:int = nondet_int()
    arr: list[int] = [1, 3, 5, 7, 9, 11]

    # Precondition
    assert is_sorted(arr)

    # --- Correctness properties ---

    # 1. If key exists, returned index is valid and points to key
    idx1: int = binary_search(arr, 7)
    assert idx1 != -1
    assert 0 <= idx1 < len(arr)
    assert arr[idx1] == 7

    # 2. Another existing element
    idx2: int = binary_search(arr, 1)
    assert idx2 != -1
    assert arr[idx2] == 1

    # 3. If key does not exist, return -1
    idx3: int = binary_search(arr, 4)
    assert idx3 == -1


if __name__ == "__main__":
    main()


