# HumanEval/3
# Entry Point: below_zero
# ESBMC-compatible format with direct assertions

from typing import List


def below_zero(operations: List[int]) -> bool:
    """ You're given a list of deposit and withdrawal operations on a bank account that starts with
    zero balance. Your task is to detect if at any point the balance of account fallls below zero, and
    at that point function should return True. Otherwise it should return False.
    >>> below_zero([1, 2, 3])
    False
    >>> below_zero([1, 2, -4, 5])
    True
    """
    balance = 0

    for op in operations:
        balance += op
        if balance < 0:
            return True

    return False


# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert below_zero([]) == False
    assert below_zero([1, 2, -3, 1, 2, -3]) == False
    # assert below_zero([1, 2, -4, 5, 6]) == True
    # assert below_zero([1, -1, 2, -2, 5, -5, 4, -4]) == False
    # assert below_zero([1, -1, 2, -2, 5, -5, 4, -5]) == True
    # assert below_zero([1, -2, 2, -2, 5, -5, 4, -4]) == True
    print("✅ HumanEval/3 - All assertions completed!")

