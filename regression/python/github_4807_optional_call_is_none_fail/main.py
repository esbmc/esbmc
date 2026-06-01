from typing import Optional


def maybe_str(flag: bool) -> Optional[str]:
    if not flag:
        return None
    return "hi"


# Negative variant: maybe_str(True) returns "hi", so `is None` must be False.
# Used to catch a regression where the simplifier returns gen_true_expr
# unconditionally for Optional[str] return values.
assert maybe_str(True) is None
