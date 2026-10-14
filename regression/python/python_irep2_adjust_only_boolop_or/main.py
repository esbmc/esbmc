# Exercises --python-irep2-adjust-only on a non-boolean short-circuit `or`.
# `len(s) or len(t)` lowers to a ternary `cond ? len(s) : len(t)` whose condition
# is the raw integer len(s) (get_truthy_condition returns a non-list value
# unchanged). With clang_cpp_adjust skipped, python_adjust's if2t arm must cast
# that condition to bool, or goto-convert rejects it: "first argument of `if'
# must be boolean". A correct verdict here proves the sole-adjuster path handles
# non-boolean BoolOp selects.
def pick_len(s: str, t: str) -> int:
    return len(s) or len(t)


assert pick_len("", "abc") == 3
assert pick_len("hi", "") == 2
assert pick_len("", "") == 0
