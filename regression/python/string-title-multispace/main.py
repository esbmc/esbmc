# Test: String title with multiple spaces
text = "sample   text"
text_title = text.title()
assert text_title == "Sample   Text"
assert text_title != "Sample   text"
