# Runtime operational model for str.count.
#
# Before this change, handle_string_count required both the receiver and
# the needle to be compile-time constant strings; non-constant arguments
# fell back to a nondet int (precise but useless) or, before the
# nondet-fallback PRs, aborted GOTO conversion entirely.
#
# __python_str_count(const char*, const char*) -- added to
# src/c2goto/library/python/string.c -- is a bounded operational model
# of str.count. The handler dispatches to it whenever both operands are
# addressable strings and start/end are at default. Symex then folds
# the call when concrete values flow in (as it does for all C library
# calls), and explores symbolically when they don't.
#
# This test pins exact-value assertions through a parameterised wrapper
# -- the only way they hold is if the runtime model actually computes
# the count rather than returning a nondet value.


def count_in(s: str, sub: str) -> int:
    return s.count(sub)


if __name__ == "__main__":
    # Receiver and needle are parameters; previously aborted/nondet.
    # Now: handler emits __python_str_count(s, sub); symex folds with
    # the concrete call-site arguments.
    assert count_in("hello", "l") == 2
    assert count_in("abcabc", "ab") == 2
    assert count_in("xxx", "y") == 0
    assert count_in("aaaa", "aa") == 2     # non-overlapping
    assert count_in("hello", "") == 6      # empty needle: len(s)+1
