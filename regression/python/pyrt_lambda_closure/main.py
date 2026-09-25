# The lambda reads n from the function around it. A function here has nowhere
# to keep a captured environment, and binding n to a same-named global instead
# would answer wrongly, so this is refused.
def outer():
    n = 5
    f = lambda x: x + n
    return f(1)


assert outer() == 6
