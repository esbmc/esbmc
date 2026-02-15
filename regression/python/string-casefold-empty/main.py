# Test: String casefold on empty string
empty = ""
empty_folded = empty.casefold()
assert empty_folded == ""
assert empty_folded != " "
