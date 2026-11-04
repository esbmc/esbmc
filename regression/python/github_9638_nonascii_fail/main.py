# github issue 9638: CPython classifies code points, so both hold.
d = "٣"
t = "ǅ"
assert not d.isdecimal() or not t.istitle()
