# Negative companion to github_4807_tuple_return_locals: the fix lets
# the function body emit so an assertion that contradicts the actual
# tuple value reaches the solver and fails (rather than aborting in
# conversion).


def f():
    a = 5
    b = 7
    return (a, b)


if __name__ == "__main__":
    assert f() == (1, 2)  # must fail: actual return is (5, 7)
