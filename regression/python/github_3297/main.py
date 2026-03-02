def my_find(s: str, c: str, start_index: int = 0) -> int:
    """Find the index of the first occurrence of a character in a string."""
    for i in range(start_index, len(s)):
        if s[i] == c:
            return i
    return -1


s: str = "foo:bar"
i = my_find(s, ':')
assert i == 3
