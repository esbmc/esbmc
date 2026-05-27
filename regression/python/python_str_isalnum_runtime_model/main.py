# Runtime operational model for str.isalnum.
#
# Before this change, handle_string_isalnum required a compile-time
# constant receiver; non-constant receivers fell back to a nondet bool
# (useless for proving anything specific).
#
# __python_str_isalnum(const char *) -- added to
# src/c2goto/library/python/string.c -- is a bounded ASCII predicate
# that returns True iff the string is non-empty and every character is
# a letter or digit. Matches Python's behaviour on the ASCII subset;
# broader Unicode coverage is deliberately out of scope, consistent
# with the existing isalpha/isdigit/isspace models.
#
# Pins exact-value assertions through a parameterised wrapper -- only
# pass if the model actually computes the predicate.


def is_alnum(s: str) -> bool:
    return s.isalnum()


if __name__ == "__main__":
    assert is_alnum("abc123") == True
    assert is_alnum("ABC") == True
    assert is_alnum("123") == True
    assert is_alnum("abc 123") == False   # space breaks it
    assert is_alnum("") == False          # Python: empty -> False
    assert is_alnum("hello!") == False    # punctuation
    assert is_alnum("_var") == False      # underscore is NOT alnum
