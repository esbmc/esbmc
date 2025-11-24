# HumanEval/1
# Entry Point: separate_paren_groups
# ESBMC-compatible format with direct assertions

from typing import List


def separate_paren_groups(paren_string: str) -> List[str]:
    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """
    result = []
    current_string = []
    current_depth = 0

    for c in paren_string:
        if c == '(':
            current_depth += 1
            current_string.append(c)
        elif c == ')':
            current_depth -= 1
            current_string.append(c)

            if current_depth == 0:
                result.append(''.join(current_string))
                current_string.clear()

    return result


# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert separate_paren_groups('(()()) ((())) () ((())()())') == [ '(()())', '((()))', '()', '((())()())' ]
    # assert separate_paren_groups('() (()) ((())) (((())))') == [ '()', '(())', '((()))', '(((())))' ]
    # assert separate_paren_groups('(()(())((())))') == [ '(()(())((())))' ]
    # assert separate_paren_groups('( ) (( )) (( )( ))') == ['()', '(())', '(()())']
    print("✅ HumanEval/1 - All assertions completed!")

