s = "hello"
assert s[:None] == "hello"
assert s[None:] == "hello"
assert s[None:None:-1] == "olleh"
assert s[None:2] == "he"
assert len(s[1:None:None]) == 4
