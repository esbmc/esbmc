# Negative variant: an uncaught exception under --irep2-bodies must be FAILED.
# abs() on a str raises TypeError with nothing to catch it. Before the throw
# round-trip fix, the body round-trip lowered the side_effect("cpp-throw") to a
# code form that convert_expression dropped, yielding a false SUCCESSFUL; this
# test guards that regression.
x = abs("hello")
