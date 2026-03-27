# Test list comprehension with function call
# Verifies that function calls in list comprehension are properly materialized


def increment(x: int) -> int:
    return x + 1


def double(x: int) -> int:
    return x * 2


if __name__ == "__main__":
    nums = [1, 2, 3]

    # Test 1
    res1 = [increment(x) for x in nums]
    assert res1[0] == 2
    assert res1[1] == 3
    assert res1[2] == 4

    # Test 2
    res2 = [double(x) for x in nums]
    assert res2[0] == 2
    assert res2[1] == 4
    assert res2[2] == 6
