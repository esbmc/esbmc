# Regression for #5499 (and the #5495 hoist): str % formatting handling now runs
# in get_binary_operator_expr before the is_array() gate, so it covers both char
# arrays (literals) and char pointers (str variables). Literal format + constant
# args still fold to the correct string here.
def labelled() -> str:
    return "item-%d" % 7


def main() -> None:
    assert labelled() == "item-7"
    assert "%s=%d" % ("x", 5) == "x=5"


main()
