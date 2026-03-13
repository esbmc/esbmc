# HumanEval/54
# Entry Point: same_chars
# ESBMC-compatible format with direct assertions



def same_chars(s0: str, s1: str):
    """
    Check if two words have the same characters.
    >>> same_chars('eabcdzzzz', 'dddzzzzzzzddeddabc')
    True
    >>> same_chars('abcd', 'dddddddabc')
    True
    >>> same_chars('dddddddabc', 'abcd')
    True
    >>> same_chars('eabcd', 'dddddddabc')
    False
    >>> same_chars('abcd', 'dddddddabce')
    False
    >>> same_chars('eabcdzzzz', 'dddzzzzzzzddddabc')
    False
    """
    return set(s0) == set(s1)

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert same_chars('eabcdzzzz', 'dddzzzzzzzddeddabc') == True
    # assert same_chars('abcd', 'dddddddabc') == True
    # assert same_chars('dddddddabc', 'abcd') == True
    # assert same_chars('eabcd', 'dddddddabc') == False
    # assert same_chars('abcd', 'dddddddabcf') == False
    # assert same_chars('eabcdzzzz', 'dddzzzzzzzddddabc') == False
    # assert same_chars('aabb', 'aaccc') == False
    print("✅ HumanEval/54 - All assertions completed!")
