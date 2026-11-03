# Issue #9080: an exception raised in a try's else clause is not covered by
# that try's own handlers, so this ValueError is uncaught.
try:
    y = 1
except ValueError:
    pass
else:
    raise ValueError("e")
