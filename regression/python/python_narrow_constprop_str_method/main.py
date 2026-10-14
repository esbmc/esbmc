# Narrow AST-level constant propagation for str method receivers.
#
# Before this change, try_extract_const_string_expr only resolved a Name
# receiver via the symbol table's cached value. Function-local string
# literals didn't have that value materialised at conversion time, so
# `s = "hello"; s.count('l')` had to be statically deemed non-foldable
# and fell back to the nondet path -- losing precision on a clearly
# constant case.
#
# The new fallback walks the current function's AST: if the variable
# has exactly one assignment AND that assignment's value is a string
# Constant, the literal is propagated. Deliberately narrow: no
# enclosing-scope walk, no aliasing, no AugAssign -- we just unlock the
# canonical local-string-literal pattern without taking on a real
# const-prop pass's soundness obligations.


def via_assign():
    s = "hello"
    return s.count('l')


def via_ann_assign():
    s: str = "hello"
    return s.count('l')


if __name__ == "__main__":
    # Exact return values -- only pass if the constant was propagated
    # into the count() handler. Without propagation, count() returns a
    # nondet int and the equality below would be falsifiable.
    assert via_assign() == 2
    assert via_ann_assign() == 2
