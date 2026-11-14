# GitHub #9148: an and/or tail is evaluated conditionally, so its guard is
# planted in the operand's own block.
if nondet_bool():
    r = "a"
else:
    r = 1
assert not isinstance(r, str) or r.upper() == "B"
