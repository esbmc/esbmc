# A callable returned by a function without a return annotation: the frontend
# types the value None (bool*), so the indirect call needs the generic
# function-pointer cast the adjuster relies on. Without it ESBMC aborted on
# to_code_type's `type.id() == typet::t_code` (#6640).
def inc(m: int) -> int:
    return m + 1


def century(m: int) -> int:
    return m + 100


def pick(c: bool):
    return inc if c else century


first = pick(True)
assert first(1) == 2

second = pick(False)
assert second(1) == 101
