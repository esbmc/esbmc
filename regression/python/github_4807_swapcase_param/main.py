# Regression for #4807: `s.swapcase()` on a function-parameter receiver used
# to abort GOTO conversion with "swapcase() requires constant string". Two
# fixes combine here:
#  (1) python_consteval now folds str.swapcase() / isupper() / count() etc.
#      on STRING-kind PyConstValues, so call sites with constant args are
#      fully evaluated at conversion time.
#  (2) handle_string_swapcase falls back to a nondet `char *` instead of
#      aborting when its receiver is not a compile-time constant, so the
#      function body still converts (sound over-approximation).


def flip_case(s: str) -> str:
    return s.swapcase()


if __name__ == "__main__":
    # consteval folds these whole calls — assertion is checked at the
    # call site without touching the function body's swapcase().
    assert flip_case('') == ''
    assert flip_case('Hello') == 'hELLO'
    assert flip_case('aB') == 'Ab'
