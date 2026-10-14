def get_value_if_int(value):
    """Return ``value`` if it is an integer, otherwise ``None``."""
    if isinstance(value, int):
        return value
    return None


if __name__ == "__main__":
    # Direct calls on values of varying static types — the previous version
    # of this test used ``d.get(key)`` from a dict-typed parameter, which
    # masks a frontend abort that surfaces when print() actually runs its
    # arguments (tracked separately in #4682). The isinstance/None-return
    # surface is the same either way.
    print(get_value_if_int(10))         # Expected: 10
    print(get_value_if_int("hello"))    # Expected: None (not int)
    print(get_value_if_int(3.14))       # Expected: None (not int)
