# Exercises the __python_str_lstrip_chars operational model through a
# non-constant (function-parameter) string, so the call is dispatched to the
# runtime model instead of being constant-folded. Guards issue #4828: the model
# must be present in the auto-generated allowlist, otherwise the dispatched call
# would silently return a nondeterministic value.
def lstrip_x(s: str) -> str:
    return s.lstrip("x")


def main() -> int:
    result = lstrip_x("xxhello")
    assert result == "hello"
    return 0


main()
