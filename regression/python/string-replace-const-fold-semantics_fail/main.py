s = "aaa"

# replace() rescans after each match, so "aa" matches once, not twice.
assert s.replace("aa", "b") == "bb"
