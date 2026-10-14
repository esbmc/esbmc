# Runtime operational model for str.isupper.
#
# Before this change, handle_string_isupper required the receiver to be
# a compile-time constant; non-constant receivers fell back to a nondet
# bool (precise but useless for proving anything).
#
# __python_str_isupper(const char *) -- added to
# src/c2goto/library/python/string.c -- is a bounded predicate that
# mirrors __python_str_islower with the cases swapped. The handler now
# dispatches to it for non-constant receivers; symex's existing CP
# folds the call on concrete arguments.
#
# Pins exact-value assertions through a parameterised wrapper -- only
# pass if the model actually computes the predicate.


def is_upper(s: str) -> bool:
    return s.isupper()


if __name__ == "__main__":
    assert is_upper("HELLO") == True
    assert is_upper("Hello") == False
    assert is_upper("hello") == False
    assert is_upper("ABC") == True
    assert is_upper("") == False          # Python: empty -> False
    assert is_upper("123") == False       # no cased chars
    assert is_upper("ABC123") == True     # cased chars all upper
