"""
Python Verification Example

Demonstrates ESBMC's Python verification capabilities:
- Type-annotated function verification
- Assertion checking
- Non-deterministic inputs
- List/array operations

Run with: esbmc python-verify.py --unwind 10
"""

# ESBMC Python intrinsics (available when running under ESBMC)
# In real code, these would be imported from esbmc module
# For standalone testing, we provide stubs

try:
    from esbmc import nondet_int, nondet_float, nondet_bool, assume, esbmc_assert
except ImportError:
    # Stubs for running outside ESBMC
    import random
    def nondet_int() -> int:
        return random.randint(-1000, 1000)
    def nondet_float() -> float:
        return random.uniform(-1000, 1000)
    def nondet_bool() -> bool:
        return random.choice([True, False])
    def assume(cond: bool) -> None:
        if not cond:
            raise ValueError("Assumption violated")
    def esbmc_assert(cond: bool, msg: str) -> None:
        assert cond, msg


# ============================================
# Example 1: Simple Function Verification
# ============================================

def absolute_value(x: int) -> int:
    """Compute absolute value."""
    if x < 0:
        return -x
    return x


def test_absolute_value() -> None:
    """Verify absolute value properties."""
    x: int = nondet_int()
    assume(x > -1000 and x < 1000)  # Avoid overflow

    result = absolute_value(x)

    # Property: result is always non-negative
    esbmc_assert(result >= 0, "Absolute value is non-negative")

    # Property: result equals x or -x
    esbmc_assert(result == x or result == -x, "Result is x or -x")


# ============================================
# Example 2: List Operations
# ============================================

def find_max(lst: list[int]) -> int:
    """Find maximum element in a list."""
    esbmc_assert(len(lst) > 0, "List must not be empty")

    max_val: int = lst[0]
    for elem in lst:
        if elem > max_val:
            max_val = elem

    return max_val


def test_find_max() -> None:
    """Verify find_max correctness."""
    # Create symbolic list
    size: int = nondet_int()
    assume(size > 0 and size <= 5)

    lst: list[int] = []
    for _ in range(size):
        val: int = nondet_int()
        assume(val > -100 and val < 100)
        lst.append(val)

    result = find_max(lst)

    # Property: result is in the list
    found: bool = False
    for elem in lst:
        if elem == result:
            found = True
    esbmc_assert(found, "Max value is in the list")

    # Property: no element is greater than result
    for elem in lst:
        esbmc_assert(elem <= result, "No element exceeds max")


# ============================================
# Example 3: Binary Search Verification
# ============================================

def binary_search(arr: list[int], target: int) -> int:
    """
    Binary search in a sorted array.
    Returns index if found, -1 otherwise.
    """
    if len(arr) == 0:
        return -1

    left: int = 0
    right: int = len(arr) - 1

    while left <= right:
        # Invariants
        esbmc_assert(left >= 0, "Left bound non-negative")
        esbmc_assert(right < len(arr), "Right bound in range")

        mid: int = left + (right - left) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1


def test_binary_search() -> None:
    """Verify binary search correctness."""
    # Create sorted array
    size: int = nondet_int()
    assume(size > 0 and size <= 5)

    arr: list[int] = []
    prev: int = nondet_int()
    assume(prev > -50)

    for _ in range(size):
        val: int = nondet_int()
        assume(val >= prev and val < prev + 10)  # Ensure sorted, bounded gaps
        arr.append(val)
        prev = val

    # Search for a value
    target: int = nondet_int()
    result = binary_search(arr, target)

    # Property: if found, element at index equals target
    if result >= 0:
        esbmc_assert(result < len(arr), "Index in bounds")
        esbmc_assert(arr[result] == target, "Found element matches target")


# ============================================
# Example 4: Safe Division
# ============================================

def safe_divide(a: int, b: int) -> int:
    """Safe integer division with zero check."""
    if b == 0:
        return 0
    return a // b


def test_safe_divide() -> None:
    """Verify safe division never crashes."""
    a: int = nondet_int()
    b: int = nondet_int()

    # No assumptions needed - function handles all cases
    result = safe_divide(a, b)

    # Property: function always returns (doesn't crash)
    esbmc_assert(True, "Division completed")

    # Property: if b != 0, result * b is close to a
    if b != 0:
        # Note: integer division truncates
        esbmc_assert(result * b <= a, "Division lower bound")


# ============================================
# Example 5: Factorial with Bounds
# ============================================

def factorial(n: int) -> int:
    """Compute factorial with precondition."""
    esbmc_assert(n >= 0, "Input must be non-negative")
    esbmc_assert(n <= 12, "Input must be <= 12 to avoid overflow")

    if n <= 1:
        return 1

    result: int = 1
    for i in range(2, n + 1):
        result = result * i

    return result


def test_factorial() -> None:
    """Verify factorial properties."""
    n: int = nondet_int()
    assume(n >= 0 and n <= 10)

    result = factorial(n)

    # Property: factorial is always positive
    esbmc_assert(result > 0, "Factorial is positive")

    # Property: factorial(n) >= n for n >= 1
    if n >= 1:
        esbmc_assert(result >= n, "Factorial >= input")


# ============================================
# Example 6: String Operations
# ============================================

def is_palindrome(s: str) -> bool:
    """Check if string is a palindrome."""
    n: int = len(s)
    for i in range(n // 2):
        if s[i] != s[n - 1 - i]:
            return False
    return True


def test_palindrome() -> None:
    """Verify palindrome check."""
    # Test with known palindrome
    assert is_palindrome("radar") == True
    assert is_palindrome("hello") == False
    assert is_palindrome("") == True
    assert is_palindrome("a") == True

    esbmc_assert(True, "Palindrome tests passed")


# ============================================
# Example 7: Stack Implementation
# ============================================

class Stack:
    """Simple stack implementation."""

    def __init__(self, capacity: int) -> None:
        self.capacity: int = capacity
        self.items: list[int] = []

    def is_empty(self) -> bool:
        return len(self.items) == 0

    def is_full(self) -> bool:
        return len(self.items) >= self.capacity

    def push(self, item: int) -> bool:
        if self.is_full():
            return False
        self.items.append(item)
        return True

    def pop(self) -> int:
        esbmc_assert(not self.is_empty(), "Cannot pop from empty stack")
        return self.items.pop()

    def size(self) -> int:
        return len(self.items)


def test_stack() -> None:
    """Verify stack operations."""
    capacity: int = nondet_int()
    assume(capacity > 0 and capacity <= 5)

    stack = Stack(capacity)
    esbmc_assert(stack.is_empty(), "New stack is empty")

    # Push some elements
    num_ops: int = nondet_int()
    assume(num_ops >= 0 and num_ops <= 3)

    for _ in range(num_ops):
        val: int = nondet_int()
        if not stack.is_full():
            stack.push(val)

    # Size invariant
    esbmc_assert(stack.size() >= 0, "Size non-negative")
    esbmc_assert(stack.size() <= capacity, "Size within capacity")

    # Pop if not empty
    if not stack.is_empty():
        old_size: int = stack.size()
        stack.pop()
        esbmc_assert(stack.size() == old_size - 1, "Pop decreases size")


# ============================================
# Example 8: Sorting Verification
# ============================================

def is_sorted(arr: list[int]) -> bool:
    """Check if array is sorted in ascending order."""
    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            return False
    return True


def bubble_sort(arr: list[int]) -> list[int]:
    """Simple bubble sort implementation."""
    result: list[int] = arr.copy()
    n: int = len(result)

    for i in range(n):
        for j in range(0, n - i - 1):
            if result[j] > result[j + 1]:
                # Swap
                temp: int = result[j]
                result[j] = result[j + 1]
                result[j + 1] = temp

    return result


def test_bubble_sort() -> None:
    """Verify bubble sort correctness."""
    size: int = nondet_int()
    assume(size >= 0 and size <= 4)

    arr: list[int] = []
    for _ in range(size):
        val: int = nondet_int()
        assume(val > -50 and val < 50)
        arr.append(val)

    sorted_arr = bubble_sort(arr)

    # Property: result is sorted
    esbmc_assert(is_sorted(sorted_arr), "Result is sorted")

    # Property: same length
    esbmc_assert(len(sorted_arr) == len(arr), "Length preserved")


# ============================================
# Main: Run all tests
# ============================================

def main() -> None:
    """Run all verification tests."""
    test_absolute_value()
    test_find_max()
    test_binary_search()
    test_safe_divide()
    test_factorial()
    test_palindrome()
    test_stack()
    test_bubble_sort()

    print("All tests passed!")


if __name__ == "__main__":
    main()
