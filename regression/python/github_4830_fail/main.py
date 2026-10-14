# Negative variant of github_4830: the element value is correctly
# resolved to int, so a wrong expected result must be caught.
def f(items, x):
    return x - items[0][0]


assert f([[3]], 1) == 999
