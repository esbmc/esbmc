# Test: String swapcase on uppercase string
text = "SAMPLE TEXT"
text_swapped = text.swapcase()
assert text_swapped == "sample text"
assert text_swapped != "Sample Text"
