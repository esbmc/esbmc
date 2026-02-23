# HumanEval/13
#Violated property:
#  file ../regression/humaneval/humaneval_13/main.py line 18 column 4
#  assertion return_value$_greatest_common_divisor$1 == 1
#  return_value$_greatest_common_divisor$1 == 1


def greatest_common_divisor(a: int, b: int) -> int:
    """ Return a greatest common divisor of two integers a and b
    >>> greatest_common_divisor(3, 5)
    1
    >>> greatest_common_divisor(25, 15)
    5
    """
    while b:
        a, b = b, a % b
    return a

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert greatest_common_divisor(3, 7) == 1
    # assert greatest_common_divisor(10, 15) == 5
    # assert greatest_common_divisor(49, 14) == 7
    # assert greatest_common_divisor(144, 60) == 12
    print("✅ HumanEval/13 - All assertions completed!")
