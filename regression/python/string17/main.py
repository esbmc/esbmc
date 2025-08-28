# Basic identity function
def identity_string(x: str) -> str:
    return x

# Identity function with different parameter name
def pass_through(s: str) -> str:
    return s

# Non-identity function (should not be optimized)
# def modify_string(x: str) -> str:
#     return x + "!"

# Identity function with multiple parameters (returns first)
def first_param(a: str, b: str) -> str:
    return a

# Identity function with multiple parameters (returns second)
def second_param(a: str, b: str) -> str:
    return b

empty_str = ""
assert identity_string(empty_str) == empty_str
assert empty_str == identity_string(empty_str)
assert identity_string("") == ""

short_str = "hello"
assert identity_string(short_str) == short_str
assert short_str == identity_string(short_str)

spaced_str = "Hello, World!"
assert identity_string(spaced_str) == spaced_str
assert spaced_str == identity_string(spaced_str)

num_str = "Test123"
assert identity_string(num_str) == num_str
assert num_str == identity_string(num_str)

special_str = "Test@#$%^&*()"
assert identity_string(special_str) == special_str
assert special_str == identity_string(special_str)

long_str = "This is a very long string that tests the string comparison functionality with many characters to ensure proper handling"
assert identity_string(long_str) == long_str
assert long_str == identity_string(long_str)

escaped_str = "Line1\nLine2\tTabbed"
assert identity_string(escaped_str) == escaped_str
assert escaped_str == identity_string(escaped_str)

test_str = "same_content"
assert pass_through(test_str) == test_str
assert test_str == pass_through(test_str)

comp_str = "compare_this"
assert identity_string(comp_str) == pass_through(comp_str)
assert pass_through(comp_str) == identity_string(comp_str)

neq_str = "not_equal_test"
assert not (identity_string(neq_str) != neq_str)
assert not (neq_str != identity_string(neq_str))

multi_str1 = "first"
multi_str2 = "second"
assert first_param(multi_str1, multi_str2) == multi_str1
assert multi_str1 == first_param(multi_str1, multi_str2)

literal_test = "literal"
assert identity_string(literal_test) == "literal"
assert "literal" == identity_string(literal_test)

repeated_str = "aaaaaaa"
assert identity_string(repeated_str) == repeated_str
assert repeated_str == identity_string(repeated_str)

mixed_str = "MiXeD CaSe"
assert identity_string(mixed_str) == mixed_str
assert mixed_str == identity_string(mixed_str)

quoted_str = 'String with "quotes" inside'
assert identity_string(quoted_str) == quoted_str
assert quoted_str == identity_string(quoted_str)

chain_str = "chain_test"
assert identity_string(chain_str) == pass_through(chain_str) == chain_str

print("All string comparison tests passed!")

