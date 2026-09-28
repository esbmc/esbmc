# Exercises the --python-irep2-adjust-only call-return conversion. `n = len(xs)`
# binds the model's `unsigned long` return to a `signed long` variable, so the
# result must be converted at the assignment; otherwise convert_assign's
# call-valued-rhs special case hands the lhs straight to do_function_call and the
# signed variable holds an unsigned value with no cast in between.
def count_up(xs: list) -> int:
    n = len(xs)
    i = 0
    while i < n:
        i = i + 1
    return i


assert count_up([1, 2, 3]) == 3
assert count_up([]) == 0
