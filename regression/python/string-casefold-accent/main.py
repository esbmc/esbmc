# Test: String casefold with accent
text = "ÁÉÍ"
text_folded = text.casefold()
assert text_folded == "áéí"
assert text_folded != "ÁÉÍ"
