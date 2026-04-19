# Test: String title with accent-like name
text = "cafe au lait"
text_title = text.title()
assert text_title == "Cafe Au Lait"
assert text_title != "CAFE AU LAIT"
