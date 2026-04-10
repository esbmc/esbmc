def pick_len(s, t):
    return len(s) or len(t)


assert pick_len("", "abc") == 3
assert pick_len("hi", "") == 2
assert pick_len("", "") == 0
