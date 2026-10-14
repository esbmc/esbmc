# Negative variant: a symbolic prefix that is not actually a prefix must make
# startswith() return False, so the assertion fails (no spurious SUCCESSFUL).


def has_prefix(s: str, prefix: str) -> bool:
    return s.startswith(prefix)


assert has_prefix("hello", "xy")
