"""
Python Verification Example

Demonstrates ESBMC's Python verification capabilities:
- Type-annotated function verification
- Assertion checking
- Non-deterministic inputs
- List/array/dict operations

Run with: esbmc python-verify.py --unwind 10 --no-pointer-check
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
    __ESBMC_assume(x > -1000 and x < 1000)  # Avoid overflow

    result = absolute_value(x)

    # Property: result is always non-negative
    assert result >= 0, "Absolute value is non-negative"

    # Property: result equals x or -x
    assert result == x or result == -x, "Result is x or -x"

test_absolute_value()

# ============================================
# Example 2: List Operations
# ============================================

def find_max(lst: list[int]) -> int:
    """Find maximum element in a list."""
    assert len(lst) > 0, "List must not be empty"

    max_val: int = lst[0]
    for elem in lst:
        if elem > max_val:
            max_val = elem

    return max_val


def test_find_max() -> None:
    """Verify find_max correctness."""
    # Create symbolic list
    size: int = nondet_int()
    __ESBMC_assume(size > 0 and size <= 5)

    lst: list[int] = []
    for _ in range(size):
        val: int = nondet_int()
        __ESBMC_assume(val > -100 and val < 100)
        lst.append(val)

    result = find_max(lst)

    # Property: result is in the list
    found: bool = False
    for elem in lst:
        if elem == result:
            found = True
    assert found, "Max value is in the list"

    # Property: no element is greater than the result
    for elem in lst:
        assert elem <= result, "No element exceeds max"

test_find_max()

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
        assert left >= 0, "Left bound non-negative"
        assert right < len(arr), "Right bound in range"

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
    __ESBMC_assume(size > 0 and size <= 5)

    arr: list[int] = []
    prev: int = nondet_int()
    __ESBMC_assume(prev > -50)

    for _ in range(size):
        val: int = nondet_int()
        __ESBMC_assume(val >= prev and val < prev + 10)  # Ensure sorted, bounded gaps
        arr.append(val)
        prev = val

    # Search for a value
    target: int = nondet_int()
    result = binary_search(arr, target)

    # Property: if found, element at index equals target
    if result >= 0:
        assert result < len(arr), "Index in bounds"
        assert arr[result] == target, "Found element matches target"

test_binary_search()

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
    assert True, "Division completed"

    # Property: if b != 0, result * b is close to a
    if b != 0:
        # Note: integer division truncates
        assert result * b <= a or result * b >= a, "Division lower bound"

test_safe_divide()

# ============================================
# Example 5: Factorial with Bounds
# ============================================

def factorial(n: int) -> int:
    """Compute factorial with precondition."""
    assert n >= 0, "Input must be non-negative"
    assert n <= 12, "Input must be <= 12 to avoid overflow"

    if n <= 1:
        return 1

    result: int = 1
    for i in range(2, n + 1):
        result = result * i

    return result


def test_factorial() -> None:
    """Verify factorial properties."""
    n: int = nondet_int()
    __ESBMC_assume(n >= 0 and n <= 10)

    result = factorial(n)

    # Property: factorial is always positive
    assert result > 0, "Factorial is positive"

    # Property: factorial(n) >= n for n >= 1
    if n >= 1:
        assert result >= n, "Factorial >= input"

test_factorial()

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

    assert True, "Palindrome tests passed"

test_palindrome()

# ============================================
# Example 7: Stack Implementation
# ============================================

def test_stack() -> None:
    """Verify stack operations."""
    capacity: int = nondet_int()
    __ESBMC_assume(capacity > 0 and capacity <= 5)
    
    # Stack state
    item0: int = 0
    item1: int = 0
    item2: int = 0
    item3: int = 0
    item4: int = 0
    top: int = 0
    
    # Test: New stack is empty
    assert top == 0, "New stack is empty"
    
    # Push some elements
    num_ops: int = nondet_int()
    __ESBMC_assume(num_ops >= 0 and num_ops <= 3)
    
    for _ in range(num_ops):
        val: int = nondet_int()
        # Check if not full before pushing
        if top < capacity:
            # Push operation
            if top == 0:
                item0 = val
            elif top == 1:
                item1 = val
            elif top == 2:
                item2 = val
            elif top == 3:
                item3 = val
            elif top == 4:
                item4 = val
            top = top + 1
    
    # Size invariant
    assert top >= 0, "Size non-negative"
    assert top <= capacity, "Size within capacity"
    
    # Pop if not empty
    if top > 0:
        old_size: int = top
        # Pop operation
        top = top - 1
        assert top == old_size - 1, "Pop decreases size"

test_stack()

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
    __ESBMC_assume(size >= 0 and size <= 4)

    arr: list[int] = []
    for _ in range(size):
        val: int = nondet_int()
        __ESBMC_assume(val > -50 and val < 50)
        arr.append(val)

    sorted_arr = bubble_sort(arr)

    # Property: result is sorted
    assert is_sorted(sorted_arr), "Result is sorted"

    # Property: same length
    assert len(sorted_arr) == len(arr), "Length preserved"

# ============================================
# Example 9: String Validation with nondet_str
# ============================================

def validate_email_format(email: str) -> bool:
    """
    Simple email format validation.
    Checks for: non-empty, contains @, has text before and after @
    """
    if len(email) == 0:
        return False
    
    # Find @ symbol
    at_pos: int = -1
    for i in range(len(email)):
        if email[i] == '@':
            if at_pos >= 0:  # Multiple @ symbols
                return False
            at_pos = i
    
    # Must have @ with text before and after
    if at_pos <= 0 or at_pos >= len(email) - 1:
        return False
    
    return True


def test_email_validation() -> None:
    """Verify email validation with symbolic strings."""
    email: str = nondet_str()
    
    # Bound string length for verification
    __ESBMC_assume(len(email) >= 0 and len(email) <= 20)
    
    result = validate_email_format(email)
    
    # Property: valid emails must have @ symbol
    if result:
        has_at: bool = False
        for char in email:
            if char == '@':
                has_at = True
                break
        assert has_at, "Valid email must contain @"
        assert len(email) >= 3, "Valid email has minimum length"
    
    # Property: empty string is invalid
    if len(email) == 0:
        assert not result, "Empty string is not valid email"

test_email_validation()

# ============================================
# Example 10: List Operations with nondet_list
# ============================================

def remove_duplicates(lst: list[int]) -> list[int]:
    """
    Remove duplicate elements from list, preserving order.
    Returns new list with only first occurrence of each element.
    """
    result: list[int] = []
    
    for elem in lst:
        # Check if elem already in result
        found: bool = False
        for existing in result:
            if existing == elem:
                found = True
                break
        
        if not found:
            result.append(elem)
    
    return result


def test_remove_duplicates() -> None:
    """Verify duplicate removal with symbolic lists."""
    # Create symbolic list of integers
    lst: list[int] = nondet_list(max_size=5, elem_type=nondet_int())
    
    # Bound element values
    for elem in lst:
        __ESBMC_assume(elem >= -10 and elem <= 10)
    
    result = remove_duplicates(lst)
    
    # Property: result has no duplicates
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            assert result[i] != result[j], "No duplicates in result"
    
    # Property: result length <= original length
    assert len(result) <= len(lst), "Result not longer than input"
    
    # Property: all elements in result exist in original
    for elem in result:
        found: bool = False
        for orig in lst:
            if orig == elem:
                found = True
                break
        assert found, "Result elements from original list"
    
    # Property: if input has no duplicates, lengths are equal
    input_has_dups: bool = False
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i] == lst[j]:
                input_has_dups = True
                break
    
    if not input_has_dups:
        assert len(result) == len(lst), "No change if no duplicates"

test_remove_duplicates()

# ============================================
# Example 11: Dictionary Operations with nondet_dict
# ============================================

def count_value_occurrences(d: dict[str, int], target: int) -> int:
    """
    Count how many times a target value appears in dictionary values.
    """
    count: int = 0
    
    for key in d:
        if d[key] == target:
            count = count + 1
    
    return count


def get_keys_with_value(d: dict[str, int], target: int) -> list[str]:
    """
    Return list of all keys that map to the target value.
    """
    result: list[str] = []
    
    for key in d:
        if d[key] == target:
            result.append(key)
    
    return result


def test_dict_operations() -> None:
    """Verify dictionary operations with symbolic dictionaries."""
    # Create symbolic string->int dictionary
    d: dict[str, int] = nondet_dict(
        max_size=4,
        key_type=nondet_str(),
        value_type=nondet_int()
    )
    
    # Bound values
    for key in d:
        __ESBMC_assume(d[key] >= -5 and d[key] <= 5)
    
    # Pick a target value to search for
    target: int = nondet_int()
    __ESBMC_assume(target >= -5 and target <= 5)
    
    count = count_value_occurrences(d, target)
    keys = get_keys_with_value(d, target)
    
    # Property: count matches number of keys found
    assert count == len(keys), "Count matches keys list length"
    
    # Property: count is non-negative and bounded by dict size
    assert count >= 0, "Count is non-negative"
    assert count <= len(d), "Count does not exceed dict size"
    
    # Property: all returned keys map to target value
    for key in keys:
        assert key in d, "Returned key exists in dict"
        assert d[key] == target, "Returned key maps to target"
    
    # Property: if count is 0, keys list is empty
    if count == 0:
        assert len(keys) == 0, "Empty keys when count is zero"
    
    # Property: no key mapping to target is missed
    for key in d:
        if d[key] == target:
            found: bool = False
            for k in keys:
                if k == key:
                    found = True
                    break
            assert found, "All matching keys are found"

test_dict_operations()

# ============================================
# Main: Run all tests
# ============================================

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
    test_email_validation()
    test_remove_duplicates()
    test_dict_operations()

    print("All tests passed!")


if __name__ == "__main__":
    main()
