# The bare `return` can't carry a tagged value, so it's refused explicitly
# rather than silently reverting the int/str branches to the untagged crash.
def f(cond, skip):
    if skip:
        return
    if cond:
        return 1
    else:
        return "a"


x = f(nondet_bool(), False)
assert x == 1 or x == "a"
