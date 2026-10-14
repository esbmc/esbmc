# A symbolic (parameter) prefix is lowered to char*, not a char array.
# str.startswith() must handle that without aborting (regression for the
# to_array_type assertion crash; see humaneval/29).


def has_prefix(s: str, prefix: str) -> bool:
    return s.startswith(prefix)


assert has_prefix("hello", "he")
assert not has_prefix("hello", "xy")
assert has_prefix("hello", "")        # the empty string is a prefix of anything
assert not has_prefix("hi", "hello")  # a longer prefix cannot match


# A genuinely symbolic, possibly-empty prefix must still be sound: s[:n] is a
# prefix of s for every n >= 0, including the empty slice s[:0].
def slice_is_prefix(s: str, n: int) -> bool:
    return s.startswith(s[:n])


assert slice_is_prefix("abc", 0)


# The original humaneval/29 trigger: a string method applied to the loop
# variable of a comprehension, with a symbolic predicate argument.
def filter_by_prefix(strings, prefix: str):
    return [x for x in strings if x.startswith(prefix)]


assert filter_by_prefix([], "john") == []
