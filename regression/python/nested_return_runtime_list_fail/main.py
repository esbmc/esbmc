# Negative variant of nested_return_runtime_list: same nested runtime-list
# returns, but an assertion that does not hold. Confirms the recovered return
# value flows precisely (the comparison is decided, not nondet) and a false
# claim is correctly refuted — i.e. the fix is sound, not just non-crashing.
#
# A small input string and unwind bound keep the counterexample search (which
# is markedly more expensive than the positive variant) well under the CI time
# cap while still fully unwinding the str.split() model, so the refuted
# property is the list comparison below (decided) rather than a truncated
# loop's unwinding assertion.


def f(txt: str, c: bool):
    if c:
        return txt.split()
    else:
        return txt.split(',')


assert f("a b", True) == ["x", "y"]
