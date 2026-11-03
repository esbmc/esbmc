# Calling a call's result at module level calls the function `get()`
# returns, as it does inside a function (#8474).
def f():
    return 1


def get():
    return f


b = get()()
assert b == 2
