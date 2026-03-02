def identity_string(x: str) -> str:
    return x


def pass_through(s: str) -> str:
    return s


# Test Case 13: Nested function calls
nested_str = "nested_test"
assert identity_string(pass_through(nested_str)) == nested_str
assert nested_str == identity_string(pass_through(nested_str))
