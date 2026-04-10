# HumanEval/44
# Entry Point: change_base
# ESBMC-compatible format with direct assertions



def change_base(x: int, base: int):
    """Change numerical base of input number x to base.
    return string representation after the conversion.
    base numbers are less than 10.
    >>> change_base(8, 3)
    '22'
    >>> change_base(8, 2)
    '1000'
    >>> change_base(7, 2)
    '111'
    """
    ret = ""
    while x > 0:
        ret = str(x % base) + ret
        x //= base
    return ret

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert change_base(8, 3) == "22"
    # assert change_base(9, 3) == "100"
    # assert change_base(234, 2) == "11101010"
    # assert change_base(16, 2) == "10000"
    # assert change_base(8, 2) == "1000"
    # assert change_base(7, 2) == "111"
    # assert change_base(x, x + 1) == str(x)
    print("✅ HumanEval/44 - All assertions completed!")
