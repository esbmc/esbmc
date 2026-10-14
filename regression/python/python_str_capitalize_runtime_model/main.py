# Runtime operational model for str.capitalize.
#
# Before this change, handle_string_capitalize required a compile-time
# constant receiver; non-constant receivers fell back to
# build_nondet_string_fallback (sound but unconstrained -- subsequent
# comparisons against the result could never hold exactly).
#
# __python_str_capitalize(const char *) -- added to
# src/c2goto/library/python/string.c -- is a bounded ASCII transform
# (first letter uppercased, rest lowercased; non-letter chars
# preserved at their positions). 255-char cap on the receiver, same
# as the lower/upper/swapcase models.
#
# Pins exact-value assertions through a parameterised wrapper -- only
# pass if the model actually produces the capitalised string.


def capitalize(s: str) -> str:
    return s.capitalize()


if __name__ == "__main__":
    assert capitalize("hello") == "Hello"
    assert capitalize("WORLD") == "World"
    assert capitalize("hELLO") == "Hello"
    assert capitalize("123abc") == "123abc"      # non-letter first char
    assert capitalize("aBc DeF") == "Abc def"    # whitespace preserved
    assert capitalize("") == ""
