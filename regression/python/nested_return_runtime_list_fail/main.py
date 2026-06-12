# Negative variant of nested_return_runtime_list: same nested runtime-list
# returns, but an assertion that does not hold. Confirms the recovered return
# value flows precisely (the comparison is decided, not nondet) and a false
# claim is correctly refuted — i.e. the fix is sound, not just non-crashing.


def f(txt: str, c: bool):
    if c:
        return txt.split()
    else:
        return txt.split(',')


assert f("Hello world!", True) == ["WRONG", "VALUE"]
