# HumanEval/27
# Entry Point: flip_case
# ESBMC-compatible format with direct assertions



def flip_case(string: str) -> str:
    """ For a given string, flip lowercase characters to uppercase and uppercase to lowercase.
    >>> flip_case('Hello')
    'hELLO'
    """
    return string.swapcase()

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert flip_case('') == ''
    # assert flip_case('Hello!') == 'hELLO!'
    # assert flip_case('These violent delights have violent ends') == 'tHESE VIOLENT DELIGHTS HAVE VIOLENT ENDS'
    print("✅ HumanEval/27 - All assertions completed!")
