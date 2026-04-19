# Test: String swapcase with numbers and underscore
text = "a1_b"
text_swapped = text.swapcase()
assert text_swapped == "A1_B"
assert text_swapped != "A1_b"
