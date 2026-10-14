# Nondet fallback for str.join() when its iterable arg can't be
# folded to a literal List at conversion time.
#
# Before this change, handle_str_join aborted with
# "join() argument must be a list of strings" whenever the iterable
# was anything other than a literal List or a Name resolving to one
# (e.g. a `sorted(...)`, a list comprehension, an unannotated
# parameter, or a function-call result). The two abort sites now fall
# back to a sound nondet `char *` via build_nondet_string_fallback so
# GOTO conversion proceeds. Same pattern as the str.*() handler
# fallbacks in PRs #4814 (splitlines), #4815 (partition/format), and
# the runtime-model series #4818..#4826.


def join_sorted(parts):
    # Iterable is a sorted() call -- not a List literal, not a Name
    # with a foldable initialiser. Previously: handler abort.
    return ' '.join(sorted(parts))


def join_param(parts):
    # Iterable is a function parameter -- legacy var_decl had no
    # foldable initialiser. Previously: handler abort on the second
    # error path ("join() requires a list").
    return ' '.join(parts)


if __name__ == "__main__":
    # No assertions on the join result -- only checks that conversion
    # completes. The actual returned string is a sound symbolic value
    # (nondet `char *`); precise functional results require flow-
    # sensitive type inference, which is separate work.
    pass
