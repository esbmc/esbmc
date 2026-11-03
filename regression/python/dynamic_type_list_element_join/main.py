# The element types recorded for lst would be the last branch's on both
# paths, so the rebinding is refused rather than read the other path wrongly.

cond = nondet_bool()
if cond:
    lst = [1]
else:
    lst = ["a"]
if not cond:
    assert lst[0] == "a"
    assert lst[0] != 1
