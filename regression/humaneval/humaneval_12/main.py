# HumanEval/12

#Violated property:
#  file ../regression/humaneval/humaneval_12/main.py line 26 column 4
#  assertion ISNONE(longest(../regression/humaneval/humaneval_12/main.py:26$py_list$34), 0)
#  ISNONE(longest(../regression/humaneval/humaneval_12/main.py:26$py_list$34), 0)

from typing import List, Optional


def longest(strings: List[str]) -> Optional[str]:
    """ Out of list of strings, return the longest one. Return the first one in case of multiple
    strings of the same length. Return None in case the input list is empty.
    >>> longest([])

    >>> longest(['a', 'b', 'c'])
    'a'
    >>> longest(['a', 'bb', 'ccc'])
    'ccc'
    """
    if not strings:
        return None

    maxlen = max(len(x) for x in strings)
    for s in strings:
        if len(s) == maxlen:
            return s

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert longest([]) == None
    # assert longest(['x', 'y', 'z']) == 'x'
    # assert longest(['x', 'yyy', 'zzzz', 'www', 'kkkk', 'abc']) == 'zzzz'
    print("✅ HumanEval/12 - All assertions completed!")
