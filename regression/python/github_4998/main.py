# Regression for GitHub #4998: an `or` of two calls to a multi-return function,
# used as the condition of a (string-valued) ternary, segfaulted SMT encoding.
# The non-folded call reached the boolean-operator lowering as a
# code_function_callt (a statement) and was wrapped in an assignment, surviving
# into the SSA as a code_function_call2t whose null operand crashed the encoder.
# It is now normalised to a side-effect call and verifies normally.


def check(t: str) -> bool:
    if len(t) > 0:
        return True
    return False


def f(s: str) -> str:
    return 'Yes' if check(s) or check(s) else 'No'


assert f('x') == 'Yes'
