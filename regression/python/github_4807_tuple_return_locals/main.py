# Regression for #4807: returning a tuple whose elements reference
# function-local variables used to abort GOTO conversion with
# "Variable 'X' is not defined". The cause was that
# infer_return_type_from_body ran *before* the function body was
# processed, but tuple_handler->get_tuple_expr eagerly called get_expr
# on each element; for a local-var Name the symbol wasn't in the
# table yet, so get_expr aborted.
#
# The fix restricts the pre-body tuple inference to all-Constant
# tuples and defers the rest to the post-body second-pass inference.


def f():
    a = 5
    b = 7
    return (a, b)


def g(x):
    p = x + 1
    q = x * 2
    return (p, q)


if __name__ == "__main__":
    assert f() == (5, 7)
    assert g(3) == (4, 6)
