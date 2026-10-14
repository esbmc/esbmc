"""Module docstring: a bare string-literal statement.

It lowers to a decayed string literal carrying no location of its own, which
the native body dispatcher declined -- forcing the enclosing function back to
the legacy body path.
"""


def annotated(x: int) -> int:
    """Function docstring, same shape."""
    y = x + 1
    return y


def two_statements(x: int) -> int:
    """Docstring followed by real work, so the fallback would be observable."""
    "a second bare string-literal statement"
    return annotated(x) + 1


assert annotated(1) == 2
assert two_statements(1) == 4
