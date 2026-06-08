from typing import Any

def foo(x: int) -> Any:
    if x == 4:
        return True
    elif x == 2:
        return "Any"
    else:
        return 5

# The string sibling return ("Any") must no longer abort Any inference
# (issue #2848); the numeric returns drive the inferred type and verify.
# String-value equality on the Any return path is not yet modelled at the
# symex level, so it is intentionally not asserted here.
assert foo(4) == True
assert foo(0) == 5
