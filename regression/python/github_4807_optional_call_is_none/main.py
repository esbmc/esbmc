from typing import Optional


def maybe_str(flag: bool) -> Optional[str]:
    if not flag:
        return None
    return "hi"


# Direct-call `is None` / `== None`: lhs arrives as code_function_call, not
# side_effect, so the isnone simplifier saw an operand with empty type and
# fell through to `False`. The frontend must promote the call to a value
# expression. Without the fix, both asserts below trip.
assert maybe_str(False) is None
assert maybe_str(False) == None
assert maybe_str(True) is not None
assert maybe_str(True) != None
