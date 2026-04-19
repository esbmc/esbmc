# Test: String casefold with numbers and underscore
text = "SAMPLE_1"
text_folded = text.casefold()
assert text_folded == "sample_1"
assert text_folded != "SAMPLE_1"
