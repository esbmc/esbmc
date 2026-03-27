def get_value_if_int(d: dict, key):
    """
    Returns the value associated with `key` in dictionary `d`
    only if the value is an integer. Otherwise returns None.
    """
    value = d.get(key)

    if isinstance(value, int):
        return value
    return None


if __name__ == "__main__":
    data = {"a": 10, "b": "hello", "c": 3.14}

    print(get_value_if_int(data, "a"))  # Expected: 10
    print(get_value_if_int(data, "b"))  # Expected: None (not int)
    print(get_value_if_int(data, "c"))  # Expected: None (not int)
    print(get_value_if_int(data, "missing"))  # Expected: None (key not found)
