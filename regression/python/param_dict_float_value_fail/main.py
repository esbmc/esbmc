# Negative companion to param_dict_float_value (#5501): the float values read
# through the unannotated parameter dict sum to 1.0, so asserting a wrong total
# must report VERIFICATION FAILED rather than crash during SMT encoding.
def total(g):
    s = 0.0
    for k, w in g.items():
        s += w
    return s


assert total({'a': 0.5, 'b': 0.5}) == 2.0
