# Test: String title with hyphen
text = "sample-text"
text_title = text.title()
assert text_title == "Sample-Text"
assert text_title != "Sample-text"
