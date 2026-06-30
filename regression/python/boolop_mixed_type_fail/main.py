# Negative variant: the mixed-type BoolOp converts, but the asserted
# short-circuit value is wrong.


def and_int_str(a: int) -> object:
    return a and "x"  # 0 -> 0 (falsy short-circuit)


assert and_int_str(0) == 1
