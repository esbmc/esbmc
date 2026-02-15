# Test: String casefold on lowercase string
text = "sample text"
text_folded = text.casefold()
assert text_folded == "sample text"
assert text_folded != "Sample Text"
