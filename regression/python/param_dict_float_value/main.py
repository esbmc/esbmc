# Reading float values from a dict passed through an unannotated parameter
# (#5501): the parameter g has no compile-time element type, so the value is
# erased to void*. Adding a float value read through void* to a float
# accumulator lowered to s = IEEE_ADD(s, (double)w) -- a numeric cast of a
# pointer-typed term producing an ill-sorted floating-point node. The
# preprocessor now infers the value type as float from the float accumulator,
# routing the read through the float_buf path.
def total(g):
    s = 0.0
    for k, w in g.items():
        s += w
    return s


assert total({'a': 0.5, 'b': 0.5}) == 1.0
