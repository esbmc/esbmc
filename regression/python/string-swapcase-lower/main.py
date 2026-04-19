# Test: String swapcase on lowercase string
text = "sample text"
text_swapped = text.swapcase()
assert text_swapped == "SAMPLE TEXT"
assert text_swapped != "Sample Text"
