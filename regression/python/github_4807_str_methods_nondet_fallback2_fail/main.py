# Negative companion to github_4807_str_methods_nondet_fallback2: prove the
# nondet fallback is genuinely unconstrained -- an assertion pinning the
# isidentifier() result against a non-constant receiver must fail under at
# least one model the solver picks.


def f_isidentifier(s: str) -> bool:
    return s.isidentifier()


if __name__ == "__main__":
    # Pre-#4807 batch 2: this aborted GOTO conversion. Now: conversion
    # completes, isidentifier(s) is a nondet bool, and the assertion below
    # can be falsified -- so VERIFICATION FAILED is the expected verdict.
    s = "abc"
    assert f_isidentifier(s)
