import random

def insertion_sort(arr: list[int]):
    # Traverse from 1 to len(arr)
    for i in range(1, len(arr)):
        key = arr[i]  # The current element to insert
        j: int = i - 1

        # Move elements of arr[0..i-1], that are greater than key,
        # to one position ahead of their current position
        while j > 0 and key < arr[j]: # BUG: it should be j>=0 instead of j>0 
            arr[j + 1] = arr[j]
            j -= 1

        # Place the key in its correct position
        arr[j + 1] = key

        # Assertion: prefix [0..i] should be sorted after each iteration
        k = 0
        while k < i:
            try:
                assert arr[k] <= arr[k + 1], f"Array not sorted at step {i}: {arr}"
            except AssertionError as e:
                print("Assertion failed:", e)
                return arr
            k += 1

# Example usage
x = random.randint(1,1000)
y = random.randint(1,1000)
nums = [x, y]
sorted_nums = insertion_sort(nums)

