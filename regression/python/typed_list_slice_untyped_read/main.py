l = [1, -2, -3, 41, 57]
assert sum(e for e in l[:3] if len(str(e)) <= 2) == -4
assert [str(e) for e in l[3:]] == ["41", "57"]
