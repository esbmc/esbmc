# HumanEval/160
# Entry Point: do_algebra
# ESBMC-compatible format with direct assertions


def do_algebra(operator, operand):
    """
    Given two lists operator, and operand. The first list has basic algebra operations, and 
    the second list is a list of integers. Use the two given lists to build the algebric 
    expression and return the evaluation of this expression.

    The basic algebra operations:
    Addition ( + ) 
    Subtraction ( - ) 
    Multiplication ( * ) 
    Floor division ( // ) 
    Exponentiation ( ** ) 

    Example:
    operator['+', '*', '-']
    array = [2, 3, 4, 5]
    result = 2 + 3 * 4 - 5
    => result = 9

    Note:
        The length of operator list is equal to the length of operand list minus one.
        Operand is a list of of non-negative integers.
        Operator list has at least one operator, and operand list has at least two operands.

    """
    expression = str(operand[0])
    for oprt, oprn in zip(operator, operand[1:]):
        expression+= oprt + str(oprn)
    return eval(expression)

# Direct assertions for ESBMC verification
if __name__ == "__main__":
    assert do_algebra(['**', '*', '+'], [2, 3, 4, 5]) == 37
    # assert do_algebra(['+', '*', '-'], [2, 3, 4, 5]) == 9
    # assert do_algebra(['//', '*'], [7, 3, 4]) == 8, "This prints if this assert fails 1 (good for debugging!)"
    # assert True, "This prints if this assert fails 2 (also good for debugging!)"
    print("✅ HumanEval/160 - All assertions completed!")
