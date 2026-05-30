# Negative counterpart of string-lstrip-chars-runtime-success: the model returns
# a concrete, deterministic string, so an incorrect expectation must be reported
# as a violation. Under the pre-#4828 silent-nondet fallback this assertion could
# not be relied upon. See issue #4828.
def lstrip_x(s: str) -> str:
    return s.lstrip("x")


def main() -> int:
    result = lstrip_x("xxhello")
    assert result == "xxhello"
    return 0


main()
