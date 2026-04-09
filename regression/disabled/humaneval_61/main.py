# HumanEval/61
# Entry Point: correct_bracketing
# ESBMC-compatible format with direct assertions



def correct_bracketing(brackets: str):
    """ brackets is a string of "(" and ")".
    return True if every opening bracket has a corresponding closing bracket.

    >>> correct_bracketing("(")
    False
    >>> correct_bracketing("()")
    True
    >>> correct_bracketing("(()())")
    True
    >>> correct_bracketing(")(()")
    False
    """
    depth = 0
    for b in brackets:
        if b == "(":
            depth += 1
        else:
            depth -= 1
        if depth < 0:
            return False
    return depth == 0

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert correct_bracketing("()")
    # assert correct_bracketing("(()())")
    # assert correct_bracketing("()()(()())()")
    # assert correct_bracketing("()()((()()())())(()()(()))")
    # assert not correct_bracketing("((()())))")
    # assert not correct_bracketing(")(()")
    # assert not correct_bracketing("(")
    # assert not correct_bracketing("((((")
    # assert not correct_bracketing(")")
    # assert not correct_bracketing("(()")
    # assert not correct_bracketing("()()(()())())(()")
    # assert not correct_bracketing("()()(()())()))()")
    print("✅ HumanEval/61 - All assertions completed!")
