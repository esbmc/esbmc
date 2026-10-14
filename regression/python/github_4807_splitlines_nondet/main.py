# Regression for #4807: str.splitlines() on a non-constant receiver used
# to abort GOTO conversion with "splitlines() requires constant string".
# handle_string_splitlines now falls back to an empty list, so the
# function body emits and the program reaches verification.
#
# This test wraps splitlines() in a function with a string parameter --
# previously the body alone aborted conversion, so simply emitting it is
# the regression.


def split_lines(s: str):
    return s.splitlines()


if __name__ == "__main__":
    # Constant-receiver call site still folds via the original handler:
    parts = "a\nb\nc".splitlines()
    assert len(parts) == 3
