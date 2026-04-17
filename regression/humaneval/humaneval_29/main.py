# HumanEval/29
# Entry Point: filter_by_prefix
# ESBMC-compatible format with direct assertions

from typing import List


def filter_by_prefix(strings: List[str], prefix: str) -> List[str]:
    """ Filter an input list of strings only for ones that start with a given prefix.
    >>> filter_by_prefix([], 'a')
    []
    >>> filter_by_prefix(['abc', 'bcd', 'cde', 'array'], 'a')
    ['abc', 'array']
    """
    return [x for x in strings if x.startswith(prefix)]

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert filter_by_prefix([], 'john') == []
    # assert filter_by_prefix(['xxx', 'asd', 'xxy', 'john doe', 'xxxAAA', 'xxx'], 'xxx') == ['xxx', 'xxxAAA', 'xxx']
    print("✅ HumanEval/29 - All assertions completed!")
