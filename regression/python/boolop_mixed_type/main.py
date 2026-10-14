# Mixed-type BoolOp: the operands have different concrete types, so the
# result type falls back to Any while each operand stays concrete. Building
# the short-circuit select must not assert/abort on the type-id mismatch.


def and_int_str(a: int) -> object:
    return a and "x"  # 0 -> 0 (falsy short-circuit), else -> "x"


def or_bool_int(a: int, b: int) -> object:
    return (a > 0) or b  # False -> b, True -> True


assert and_int_str(0) == 0
assert or_bool_int(-1, 9) == 9
