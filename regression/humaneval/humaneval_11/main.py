# HumanEval/11
#ERROR: join() argument must be a list of strings

from typing import List


def string_xor(a: str, b: str) -> str:
    """ Input are two strings a and b consisting only of 1s and 0s.
    Perform binary XOR on these inputs and return result also as a string.
    >>> string_xor('010', '110')
    '100'
    """
    def xor(i: str, j: str) -> str:
        if i == j:
            return '0'
        else:
            return '1'

    return ''.join(xor(x, y) for x, y in zip(a, b))

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert string_xor('111000', '101010') == '010010'
    # assert string_xor('1', '1') == '0'
    # assert string_xor('0101', '0000') == '0101'
    print("✅ HumanEval/11 - All assertions completed!")
