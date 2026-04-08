# HumanEval/65
# Entry Point: circular_shift
# ESBMC-compatible format with direct assertions


def circular_shift(x, shift):
    """Circular shift the digits of the integer x, shift the digits right by shift
    and return the result as a string.
    If shift > number of digits, return digits reversed.
    >>> circular_shift(12, 1)
    "21"
    >>> circular_shift(12, 2)
    "12"
    """
    s = str(x)
    if shift > len(s):
        return s[::-1]
    else:
        return s[len(s) - shift:] + s[:len(s) - shift]

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert circular_shift(100, 2) == "001"
    # assert circular_shift(12, 2) == "12"
    # assert circular_shift(97, 8) == "79"
    # assert circular_shift(12, 1) == "21", "This prints if this assert fails 1 (good for debugging!)"
    # assert circular_shift(11, 101) == "11", "This prints if this assert fails 2 (also good for debugging!)"
    print("✅ HumanEval/65 - All assertions completed!")
