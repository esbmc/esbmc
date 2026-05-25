# Negative companion to github_4807_partition_format_nondet: confirms the
# format() nondet fallback is genuinely unconstrained. Calling f_format
# from a context where the format string is a non-constant binding pins
# the nondet path; an assertion forcing a specific result must fail
# rather than aborting in conversion.


def f_format(fmt: str, x: int) -> str:
    return fmt.format(x)


if __name__ == "__main__":
    # Pre-#4807 batch 3: this aborted GOTO conversion. Now: conversion
    # completes; the body returns nondet char *; the assertion can be
    # falsified -- VERIFICATION FAILED is the expected verdict.
    fmt = "x={}"
    result = f_format(fmt, 7)
    assert result == "x=7"
