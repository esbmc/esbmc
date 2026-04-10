# Test: String vazia - edge cases
empty = ""
assert len(empty) == 0
assert empty == ""
assert empty + "a" == "a"
assert "a" + empty == "a"
assert empty * 5 == ""
assert empty.upper() == ""
assert empty.strip() == ""
