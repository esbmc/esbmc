# HumanEval/138
# Entry Point: is_equal_to_sum_even
# ESBMC-compatible format with direct assertions


def is_equal_to_sum_even(n):
    """Evaluate whether the given number n can be written as the sum of exactly 4 positive even numbers
    Example
    is_equal_to_sum_even(4) == False
    is_equal_to_sum_even(6) == False
    is_equal_to_sum_even(8) == True
    """
    return n%2 == 0 and n >= 8

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert is_equal_to_sum_even(4) == False
    # assert is_equal_to_sum_even(6) == False
    # assert is_equal_to_sum_even(8) == True
    # assert is_equal_to_sum_even(10) == True
    # assert is_equal_to_sum_even(11) == False
    # assert is_equal_to_sum_even(12) == True
    # assert is_equal_to_sum_even(13) == False
    # assert is_equal_to_sum_even(16) == True
    print("✅ HumanEval/138 - All assertions completed!")
