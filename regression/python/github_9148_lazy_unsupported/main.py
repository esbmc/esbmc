# GitHub #9148: an and/or tail is evaluated conditionally, so no guard can be
# planted ahead of the statement.
if nondet_bool():
    r = "a"
else:
    r = 1
assert not isinstance(r, str) or r.upper() == "A"
