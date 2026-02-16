# Test: String title with underscores
text = "sample_text"
text_title = text.title()
assert text_title == "Sample_Text"
assert text_title != "Sample_text"
