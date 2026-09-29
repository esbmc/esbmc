# "+" is a string, not a float (#4797).
def count_floats(tokens):
    n = 0
    for token in tokens:
        if isinstance(token, float):
            n += 1
    return n


assert count_floats([3.0, "+", 2.0]) == 3
